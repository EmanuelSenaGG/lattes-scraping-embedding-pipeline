import subprocess
import os
import logging
import sys
import shutil
import time
import json
import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector


logging.basicConfig(level=logging.INFO, format='%(asctime)s - [PIPELINE] - %(message)s', stream=sys.stdout)

SCRIPTLATTES_PATH = "tools/scriptLattes"
CONFIG_FILE_PATH = "../../input/pipeline.config"
OUTPUT_DIRECTORY = "/app/output"


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "postgres"),
    "database": os.getenv("DB_NAME", "lattes_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "postgres")
}


OLLAMA_URL = "https://ollama-api.ifba.edu.br/api/embed"
OLLAMA_MODEL = "qwen3-embedding:4b"


MAX_CONCURRENT_REQUESTS = 15  # Limita as requisições simultâneas 
BATCH_INSERT_SIZE = 100       # Quantidade de chunks acumulados antes de fazer o INSERT no banco

def run_pipeline():
    logging.info("Iniciando extracao Lattes...")
    
    # 1. Limpa o conteudo da pasta de saida sem deletar o diretorio em si
    if os.path.exists(OUTPUT_DIRECTORY):
        logging.info(f"Limpando arquivos antigos em {OUTPUT_DIRECTORY}...")
        for item in os.listdir(OUTPUT_DIRECTORY):
            item_path = os.path.join(OUTPUT_DIRECTORY, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                logging.warning(f"Nao foi possivel deletar {item_path}: {e}")
    else:
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    try:
        # 2. Executa o scriptLattes 
        subprocess.run(
            ["xvfb-run", "-a", "venv/bin/python3", "scriptLattes.py", CONFIG_FILE_PATH],
            cwd=SCRIPTLATTES_PATH,
            check=True
        )
        
        logging.info("Iniciando organizacao dos arquivos JSON...")
        
        # 3. Localiza a pasta json gerada dentro do output
        json_source_path = os.path.join(OUTPUT_DIRECTORY, "json")
        
        if os.path.exists(json_source_path):
            # Move todos os arquivos da pasta json para a raiz do diretorio de saida
            for filename in os.listdir(json_source_path):
                shutil.move(
                    os.path.join(json_source_path, filename), 
                    os.path.join(OUTPUT_DIRECTORY, filename)
                )
            
            # 4. Remove as sobras (como a pasta json agora vazia e arquivos html/css na raiz)
            for item in os.listdir(OUTPUT_DIRECTORY):
                item_path = os.path.join(OUTPUT_DIRECTORY, item)
                # Remove pastas (incluindo a 'json' vazia) e arquivos que nao sao .json
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                elif not item.endswith(".json"):
                    os.remove(item_path)
            
            logging.info(f"Sucesso! Arquivos JSON extraidos para '{OUTPUT_DIRECTORY}'.")
        else:
            logging.warning("Pasta JSON nao encontrada. Verifique se 'global-salvar_json = sim' esta no config.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Erro na extracao: {e}")




def setup_database():
    """Testa a conexão e cria a tabela necessária para armazenar os vetores do matching acadêmico."""
    logging.info("Verificando conexao com o banco de dados e preparando tabelas...")
    for i in range(10):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Habilita pgvector
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Cria a tabela 
            # O tipo vector sem dimensão especificada 
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lattes_chunks (
                    id SERIAL PRIMARY KEY,
                    id_lattes VARCHAR(255),
                    nome_professor VARCHAR(255),
                    chave_chunk VARCHAR(255),
                    conteudo_chunk TEXT,
                    embedding vector
                );
            """)
            # Trunca  a tabela para não duplicar dados entre as execuções
            cursor.execute("TRUNCATE TABLE lattes_chunks RESTART IDENTITY;")
            logging.info("Tabela 'lattes_chunks' truncada (zerada) com sucesso para a nova execucao.")
            
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            logging.info(f"Postgres pronto! Versao: {version[0]}")
            
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logging.warning(f"Tentativa {i+1}/10: Banco de dados ainda nao disponivel...")
            time.sleep(2)
            
    return False


def insert_batch(batch):
    """Realiza a inserção em massa no PostgreSQL."""
    if not batch:
        return

    query = """
        INSERT INTO lattes_chunks (id_lattes, nome_professor, chave_chunk, conteudo_chunk, embedding)
        VALUES %s
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        cursor = conn.cursor()
        execute_values(cursor, query, batch)
        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f"Lote de {len(batch)} chunks inserido com sucesso no banco!")
    except Exception as e:
        logging.error(f"Erro ao inserir lote no banco: {e}")


async def fetch_embedding(session, semaphore, text):
    """Requisita o vetor do modelo Ollama garantindo controle de concorrência via Semáforo."""
    payload = {
        "model": OLLAMA_MODEL,
        "input": text
    }
    
    async with semaphore:
        try:
            async with session.post(OLLAMA_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
             
                    return data.get('embeddings', [[]])[0]
                else:
                    logging.error(f"Erro da API (Status {response.status}): {await response.text()}")
                    return None
        except Exception as e:
            logging.error(f"Falha na requisicao HTTP para o Ollama: {e}")
            return None


async def process_json_file(file_path, session, semaphore):
    """Lê um currículo, separa em blocos (chunks) por chave e gera os embeddings."""
    file_chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        personal_info = data.get("informacoes_pessoais", {})
        lattes_id = personal_info.get("id_lattes", "desconhecido")
        name = personal_info.get("nome_completo", "desconhecido")
        
        for key, content in data.items():
            # IGNORA a chave de informações pessoais para não gerar vetor
            if key == "informacoes_pessoais":
                continue
                
            if not content:
                continue
            
            # Converte dicionários/listas para string para o modelo ler
            chunk_text = json.dumps(content, ensure_ascii=False)        
            vector = await fetch_embedding(session, semaphore, chunk_text)
            
            if vector:
                file_chunks.append((lattes_id, name, key, chunk_text, vector))

        test_key = "chunk_teste"
        test_text = "teste"
        test_vector = await fetch_embedding(session, semaphore, test_text)
        
        if test_vector:
            file_chunks.append((lattes_id, name, test_key, test_text, test_vector))

    except Exception as e:
        logging.error(f"Erro ao processar o arquivo {file_path}: {e}")
        
    return file_chunks


async def run_embedding_pipeline():
    """Gerencia o ciclo de vida assíncrono para processar todos os currículos Lattes."""
    logging.info("Iniciando pipeline de geração de embeddings...")
    
    json_files = [os.path.join(OUTPUT_DIRECTORY, f) for f in os.listdir(OUTPUT_DIRECTORY) if f.endswith('.json')]
    
    if not json_files:
        logging.warning("Nenhum arquivo JSON encontrado para gerar embeddings.")
        return

    # O semáforo controla quantas requisições ocorrem no mesmo instante de tempo
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    global_batch = []
    
    async with aiohttp.ClientSession() as session:
        # 1. Cria as tarefas (removido o argumento global_batch)
        tasks = [process_json_file(f, session, semaphore) for f in json_files]
        
        # 2. Executa todos os arquivos simultaneamente e aguarda os retornos
        results = await asyncio.gather(*tasks)
        
        # 3. Agrupa os resultados e faz o batch insert de forma segura
        for file_chunks in results:
            if file_chunks: # Se a lista não for vazia
                global_batch.extend(file_chunks)
            
            # Enquanto o lote global for maior ou igual a 100, insere no banco
            while len(global_batch) >= BATCH_INSERT_SIZE:
                # Pega uma fatia do tamanho do batch e insere
                batch_to_insert = global_batch[:BATCH_INSERT_SIZE]
                insert_batch(batch_to_insert)
                # Remove os itens já inseridos da lista
                global_batch = global_batch[BATCH_INSERT_SIZE:]
                
        # 4. Salva eventuais chunks que sobraram na lista (lote incompleto)
        if global_batch:
            insert_batch(global_batch)
            
    logging.info("Processamento de embeddings concluído!")



def log_professor_summary():
    """Consulta o banco de dados e exibe no log o primeiro e o último chunk de cada professor com detalhes."""
    logging.info("--- Gerando Resumo Detalhado de Registros por Professor ---")
    
    # A query agora traz o vetor de forma nativa (sem conversão para texto)
    query = """
        WITH Extremos AS (
            SELECT 
                nome_professor,
                MIN(id) as min_id,
                MAX(id) as max_id,
                COUNT(id) as total_chunks
            FROM lattes_chunks
            GROUP BY nome_professor
        )
        SELECT 
            e.nome_professor,
            e.total_chunks,
            
            -- Dados do Primeiro Chunk (min_id)
            c_min.id AS min_id,
            c_min.id_lattes AS min_lattes,
            c_min.chave_chunk AS min_chave,
            LEFT(c_min.conteudo_chunk, 60) AS min_texto,
            c_min.embedding AS min_vetor,
            
            -- Dados do Último Chunk (max_id)
            c_max.id AS max_id,
            c_max.id_lattes AS max_lattes,
            c_max.chave_chunk AS max_chave,
            LEFT(c_max.conteudo_chunk, 60) AS max_texto,
            c_max.embedding AS max_vetor
            
        FROM Extremos e
        JOIN lattes_chunks c_min ON e.min_id = c_min.id
        JOIN lattes_chunks c_max ON e.max_id = c_max.id
        ORDER BY e.nome_professor;
    """
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        # Importante: Ensina o Python a traduzir o tipo vector do Postgres para um array nativo
        register_vector(conn)
        cursor = conn.cursor()
        
        cursor.execute(query)
        resultados = cursor.fetchall()
        
        if not resultados:
            logging.warning("Nenhum registro encontrado no banco de dados.")
        else:
            for row in resultados:
                nome, total = row[0], row[1]
                min_id, min_lattes, min_chave, min_texto, min_vetor = row[2], row[3], row[4], row[5], row[6]
                max_id, max_lattes, max_chave, max_texto, max_vetor = row[7], row[8], row[9], row[10], row[11]       
                min_vetor_snippet = f"{[float(x) for x in min_vetor[:3]]}..." if min_vetor is not None else "[]"
                max_vetor_snippet = f"{[float(x) for x in max_vetor[:3]]}..." if max_vetor is not None else "[]"

                resumo_texto = (
                    f"\n======================================================\n"
                    f"Prof: {nome} | Total de Chunks: {total}\n"
                    f"  [PRIMEIRO REGISTRO]\n"
                    f"    ID Banco: {min_id} | Lattes: {min_lattes}\n"
                    f"    Chave: '{min_chave}'\n"
                    f"    Conteúdo: {min_texto}...\n"
                    f"    Vetor: {min_vetor_snippet}\n"
                    f"  [ÚLTIMO REGISTRO]\n"
                    f"    ID Banco: {max_id} | Lattes: {max_lattes}\n"
                    f"    Chave: '{max_chave}'\n"
                    f"    Conteúdo: {max_texto}...\n"
                    f"    Vetor: {max_vetor_snippet}\n"
                    f"======================================================"
                )

                logging.info(resumo_texto)
                
        cursor.close()
        conn.close()
        logging.info("--- Fim do Resumo ---")
        
    except Exception as e:
        logging.error(f"Erro ao buscar resumo dos professores no banco: {e}")


if __name__ == "__main__":
    run_pipeline()
    if setup_database():
        asyncio.run(run_embedding_pipeline())
        log_professor_summary()      
    else:
        logging.error("Falha ao configurar o banco de dados. Encerrando execução.")


