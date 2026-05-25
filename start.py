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
from psycopg2.extras import execute_values, execute_batch
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

#registros do banco em memoria para processamento mais rapido
CACHE_CHUNKS = {}

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




def update_batch(batch):
    """Realiza a atualização em massa no PostgreSQL usando o ID do banco."""
    if not batch:
        return

    query = """
        UPDATE lattes_chunks 
        SET conteudo_chunk = %s, embedding = %s 
        WHERE id = %s
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        cursor = conn.cursor()
        execute_batch(cursor, query, batch)
        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f"Lote de {len(batch)} chunks ATUALIZADO com sucesso no banco!")
    except Exception as e:
        logging.error(f"Erro ao atualizar lote no banco: {e}")




async def fetch_embedding(session, text, semaphore=None):
    """Requisita o vetor do modelo Ollama. O semáforo é opcional para chamadas isoladas."""
    payload = {
        "model": OLLAMA_MODEL,
        "input": text
    }
    
   
    async def _go_request():
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


    if semaphore:
        async with semaphore:
            return await _go_request()

    else:
        return await _go_request()


async def process_json_file(file_path, session, semaphore):

    file_data = {"inserts": [], "updates": []}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        personal_info = data.get("informacoes_pessoais", {})
        lattes_id = personal_info.get("id_lattes", "desconhecido")
        name = personal_info.get("nome_completo", "desconhecido")
        
        for key, content in data.items():
            if key == "informacoes_pessoais" or not content:
                continue
            
            chunk_text = json.dumps(content, ensure_ascii=False)
            cache_key = (lattes_id, key)
            
            # Se ja existe, verifica mudança, se mudou, adiciona a lista de updates, se nao, pula pro proximo
            if cache_key in CACHE_CHUNKS:
                saved_content = CACHE_CHUNKS[cache_key]["conteudo_chunk"]        
                if saved_content == chunk_text:
                    continue 
                else:
                    logging.info(f"Conteúdo alterado no chunk '{key}' do Lattes {lattes_id} do Professor {name}. Atualizando...")
                    vector = await fetch_embedding(session, chunk_text, semaphore)
                    if vector:
                        chunk_id = CACHE_CHUNKS[cache_key]["id"]
                        file_data["updates"].append((chunk_text, vector, chunk_id))
            
            # Se não existe então é novo, vai pra lista de insert
            else:
                vector = await fetch_embedding(session, chunk_text, semaphore)
                if vector:
                    file_data["inserts"].append((lattes_id, name, key, chunk_text, vector))

    except Exception as e:
        logging.error(f"Erro ao processar o arquivo {file_path}: {e}")
        
    return file_data


async def run_embedding_pipeline():
    """Gerencia o ciclo de vida assíncrono para processar todos os currículos Lattes."""
    logging.info("Iniciando pipeline de geração de embeddings...")
    
    json_files = [os.path.join(OUTPUT_DIRECTORY, f) for f in os.listdir(OUTPUT_DIRECTORY) if f.endswith('.json')]
    
    if not json_files:
        logging.warning("Nenhum arquivo JSON encontrado para gerar embeddings.")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    global_batch_inserts = []
    global_batch_updates = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_json_file(f, session, semaphore) for f in json_files]
        results = await asyncio.gather(*tasks)
        
        for file_data in results:
            if file_data["inserts"]:
                global_batch_inserts.extend(file_data["inserts"])
            if file_data["updates"]:
                global_batch_updates.extend(file_data["updates"])
            

            while len(global_batch_inserts) >= BATCH_INSERT_SIZE:
                insert_batch(global_batch_inserts[:BATCH_INSERT_SIZE])
                global_batch_inserts = global_batch_inserts[BATCH_INSERT_SIZE:]
                
     
            while len(global_batch_updates) >= BATCH_INSERT_SIZE:
                update_batch(global_batch_updates[:BATCH_INSERT_SIZE])
                global_batch_updates = global_batch_updates[BATCH_INSERT_SIZE:]
                
  
        if global_batch_inserts:
            insert_batch(global_batch_inserts)
        if global_batch_updates:
            update_batch(global_batch_updates)
            
    logging.info("Processamento de embeddings concluído!")



async def log_match_results():
    """Gera vetores para propostas mockadas de TCC e busca os 5 professores mais aderentes via Distancia de Cosseno."""
    logging.info("--- Iniciando Teste de Matching (Similaridade de Cosseno) ---")
    

    proposals_tcc = [

        "Desenvolvimento de um sistema inteligente de recomendação acadêmica para alocação e match de alunos e orientadores de TCC. O projeto utiliza Inteligência Artificial, Processamento de Linguagem Natural (NLP) e a técnica de Retrieval-Augmented Generation (RAG). A arquitetura envolve a geração de embeddings via LLMs, cálculo de similaridade de cosseno e busca semântica de alta performance estruturada em um banco de dados vetorial PostgreSQL utilizando a extensão pgvector.",
        
        "Análise, monitoramento e otimização de performance (Database Tuning) em sistemas de bancos de dados relacionais Microsoft SQL Server de alta concorrência. O estudo tem como foco a refatoração de stored procedures complexas e sistemas legados, otimização e manutenção de índices, tuning de queries para mitigação de gargalos de I/O, além de análise profunda de planos de execução (Execution Plans) visando garantir escalabilidade e baixa latência no processamento de transações (OLTP)."
    ]


    query = """
        SELECT 
            nome_professor,
            id_lattes,
            chave_chunk, 
            conteudo_chunk, 
            embedding <=> %s::vector AS distancia_cosseno
        FROM lattes_chunks
        ORDER BY distancia_cosseno ASC
        LIMIT 5;
    """

    for i, proposal in enumerate(proposals_tcc, 1):
        logging.info(f"\nGerando embedding para a Proposta {i}...")
        

        vector_proposal = await get_proposal_vector(proposal)
        
        if not vector_proposal:
            logging.error(f"Falha ao gerar vetor para a Proposta {i}. Pulando...")
            continue
            
        try:
           
            conn = psycopg2.connect(**DB_CONFIG)
            register_vector(conn)
            cursor = conn.cursor()
            

            cursor.execute(query, [vector_proposal])
            results = cursor.fetchall()
            
        
            log_text = f"\n================ RESULTADO DA PROPOSTA {i} ================\n"
            log_text += f"TEXTO DA PROPOSTA: '{proposal}'\n\n"
            log_text += "TOP 5 MATCHES (Ordenados por Menor Distância de Cosseno):\n"
            
            for rank, row in enumerate(results, 1):
                nome, lattes, chave, conteudo, distancia = row
                
                log_text += f"\n  [{rank}º LUGAR] - Distância: {distancia:.4f}\n"
                log_text += f"    Professor: {nome} (Lattes: {lattes})\n"
                log_text += f"    Chave Lattes: '{chave}'\n"
                log_text += f"    Conteúdo do Chunk: {conteudo[:200]}...\n"
                
            log_text += "===========================================================\n"
            logging.info(log_text)
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logging.error(f"Erro ao consultar o banco para a proposta {i}: {e}")



def load_chunks_to_memory():
    """Busca registros na tabela lattes_chunks e popula o dicionário CACHE_CHUNKS na memória."""
    global CACHE_CHUNKS
    logging.info("Carregando chunks existentes do banco para a memória...")
    
    query = "SELECT id, id_lattes, chave_chunk, conteudo_chunk FROM lattes_chunks;"
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(query)
        resultados = cursor.fetchall()
        
        for row in resultados:
            chunk_id, id_lattes, chave_chunk, conteudo_chunk = row
            
            # Usa a tupla (id_lattes, chave_chunk) como chave principal do dicionário
            CACHE_CHUNKS[(id_lattes, chave_chunk)] = {
                "id": chunk_id,
                "conteudo_chunk": conteudo_chunk
            }
            
        cursor.close()
        conn.close()
        logging.info(f"Sucesso! {len(CACHE_CHUNKS)} chunks carregados em cache (memória).")
        
    except Exception as e:
        logging.error(f"Erro ao carregar chunks para a memória: {e}")





def clear_chunks_memory():

    global CACHE_CHUNKS
    CACHE_CHUNKS.clear()
    logging.info("Cache em memória (CACHE_CHUNKS) limpo com sucesso.")


async def get_proposal_vector(proposta_text):
    async with aiohttp.ClientSession() as session:
        return await fetch_embedding(session, proposta_text)


if __name__ == "__main__":
    #run_pipeline()     
    if setup_database():
        load_chunks_to_memory()
        asyncio.run(run_embedding_pipeline())    
        clear_chunks_memory()
        asyncio.run(log_match_results())      
        
    else:
        logging.error("Falha ao configurar o banco de dados. Encerrando execução.")

