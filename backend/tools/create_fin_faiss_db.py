from typing import List
import faiss
import numpy as np
import sqlite3
import logging
import os
import time
import pandas as pd
from tqdm import tqdm
from zhipu_embedding import ZhipuAIEmbedding
from dotenv import load_dotenv
import psutil
import sys
import importlib
load_dotenv()

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 backend 目录 (脚本目录的上一级)
backend_dir = os.path.dirname(script_dir)
# 获取项目根目录 (backend 目录的上一级)
project_root = os.path.dirname(backend_dir)

# 修正后的文件和目录路径
DB_PATH = os.path.join(backend_dir, 'db')  # 数据库目录: backend/db
DATA_DIR = os.path.join(backend_dir, 'data')  # 数据目录: backend/data

# 设置日志
log_file = os.path.join(DB_PATH, 'financial_vector_db.log')
os.makedirs(DB_PATH, exist_ok=True)  # 确保日志目录存在

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def get_csv_row_count(file_path):
    """高效获取CSV文件行数，包含完善的错误处理
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        int or None: 文件行数（不包含标题行），如果出错则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 检查文件是否为空
            first_line = f.readline()
            if not first_line:
                logging.warning(f"CSV文件为空: {file_path}")
                return 0
                
            # 计算剩余行数（不包含已读取的标题行）
            row_count = sum(1 for _ in f)
            logging.info(f"CSV文件共计 {row_count} 行数据（不含标题行）")
            return row_count
            
    except FileNotFoundError:
        logging.error(f"找不到CSV文件: {file_path}")
        return None
    except PermissionError:
        logging.error(f"没有权限读取文件: {file_path}")
        return None
    except UnicodeDecodeError:
        logging.error(f"文件编码错误，请确保是UTF-8编码: {file_path}")
        return None
    except Exception as e:
        logging.error(f"读取CSV文件行数时发生错误: {str(e)}")
        return None

class FinancialVectorDB:
    def __init__(self, db_path=DB_PATH, db_name='financial_terms_zhipu'):
        """初始化金融术语向量数据库"""
        # 初始化智谱AI embedding模型
        self.embed_model = ZhipuAIEmbedding(timeout=60)
        
        # 创建必要的目录
        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path
        self.db_name = db_name
        
        # 初始化SQLite数据库
        self.sqlite_path = os.path.join(db_path, f'{db_name}.db')
        self.conn = sqlite3.connect(self.sqlite_path)
        self.create_tables()
        
        # API请求间隔控制
        self.request_interval = 1.0
        self.batch_size = 64
        
        logging.info(f"API请求间隔设置为 {self.request_interval} 秒")
        logging.info(f"批处理大小设置为 {self.batch_size}")
        
        # 创建进度记录表
        self.create_progress_table()
        
        # 尝试加载现有索引
        self.load_existing_index()

        # 检查数据库完整性
        if not self.check_database_integrity():
            logging.warning("数据库可能存在问题，建议删除后重新创建")
        
        # 确保embedding模型正确初始化
        self.initialize_embedding_model()

    def create_tables(self):
        """创建SQLite表结构"""
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS financial_terms (
                    id INTEGER PRIMARY KEY,
                    term_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    embedding_id INTEGER,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # 创建索引以提高查询性能
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_term_name ON financial_terms(term_name)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_category ON financial_terms(category)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON financial_terms(status)')

    def create_progress_table(self):
        """创建进度记录表"""
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS processing_progress (
                    id INTEGER PRIMARY KEY,
                    chunk_index INTEGER,
                    batch_start_idx INTEGER,
                    total_processed INTEGER,
                    last_update TIMESTAMP
                )
            ''')

    def get_last_progress(self):
        """获取上次处理进度，增加验证逻辑"""
        try:
            cursor = self.conn.execute('''
                SELECT chunk_index, batch_start_idx, total_processed
                FROM processing_progress
                ORDER BY last_update DESC
                LIMIT 1
            ''')
            progress = cursor.fetchone()
            
            # 增加进度记录验证
            if progress:
                chunk_index, batch_start_idx, total_processed = progress
                if chunk_index < 0 or batch_start_idx < 0 or total_processed < 0:
                    logging.warning("检测到无效的进度记录，将重新开始处理")
                    return None
            return progress
            
        except sqlite3.Error as e:
            logging.error(f"读取进度记录时发生错误: {e}")
            return None

    def update_progress(self, chunk_index, batch_start_idx, total_processed):
        """更新处理进度"""
        with self.conn:
            self.conn.execute('''
                INSERT INTO processing_progress 
                (chunk_index, batch_start_idx, total_processed, last_update)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (chunk_index, batch_start_idx, total_processed))

    def load_existing_index(self):
        """加载现有的FAISS索引"""
        index_path = os.path.join(self.db_path, f'{self.db_name}.index')
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
                logging.info(f"成功加载现有索引: {index_path}")
                self.dimension = self.index.d
                return True
            except Exception as e:
                logging.error(f"加载现有索引失败: {e}")
        return False

    def save_index(self, is_final=False):
        """保存FAISS索引
        
        Args:
            is_final (bool): 是否是最终保存，如果是则使用正式文件名，否则添加.temp后缀
        """
        try:
            index_path = os.path.join(self.db_path, f'{self.db_name}.index')
            if not is_final:
                index_path += '.temp'
            
            faiss.write_index(self.index, index_path)
            logging.info(f"已保存索引到: {index_path}")
            
            if is_final:
                # 删除临时文件
                temp_path = index_path + '.temp'
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logging.error(f"保存索引时出错: {e}")
            raise

    def initialize_index(self, file_path):
        """使用实际数据初始化FAISS索引"""
        try:
            # 读取CSV文件的第一行数据作为样本
            df_sample = pd.read_csv(file_path, dtype=str, nrows=1, header=None, names=['term_name', 'category'])
            if df_sample.empty:
                raise ValueError("CSV文件为空")
            
            # 使用实际的金融术语作为样本
            sample_text = df_sample['term_name'].iloc[0]
            logging.info(f"使用样本文本初始化: {sample_text}")
            
            # 获取向量维度
            sample_embedding = self.embed_model._get_text_embedding(sample_text)
            self.dimension = len(sample_embedding)
            logging.info(f"向量维度: {self.dimension}")
            
            # 初始化FAISS索引
            self.index = faiss.IndexFlatIP(self.dimension)
            return True
        
        except Exception as e:
            logging.error(f"初始化索引时出错: {e}")
            raise

    def adjust_request_interval(self, error_count):
        """根据错误次数动态调整请求间隔"""
        if error_count > 0:
            self.request_interval = min(5.0, self.request_interval * 1.5)
            self.batch_size = max(32, self.batch_size // 2)
            logging.warning(f"检测到API错误，调整间隔至 {self.request_interval} 秒")
            logging.warning(f"批处理大小调整至 {self.batch_size}")
        else:
            # 可以适当减少间隔，但不要低于1秒
            self.request_interval = max(1.0, self.request_interval * 0.9)

    def process_batch(self, batch_df, file_path):
        """处理单个批次的数据"""
        error_count = 0
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                docs = [row['term_name'] for _, row in batch_df.iterrows()]
                embeddings = self.embed_model._get_text_embeddings(docs)
                embeddings_np = np.array(embeddings, dtype=np.float32)
                
                # 添加到FAISS索引
                self.index.add(embeddings_np)
                
                # 批量插入SQLite
                with self.conn:
                    self.conn.executemany('''
                        INSERT INTO financial_terms (
                            term_name, category, embedding_id, status
                        ) VALUES (?, ?, ?, ?)
                    ''', [
                        (str(row['term_name']),
                         str(row['category']),
                         idx,
                         'active') for idx, (_, row) in enumerate(batch_df.iterrows())
                    ])
                
                break
            except Exception as e:
                error_count += 1
                logging.error(f"批处理错误 (尝试 {retry+1}/{max_retries}): {e}")
                self.adjust_request_interval(error_count)
                if retry < max_retries - 1:
                    time.sleep(self.request_interval * 2)  # 错误后等待更长时间
                else:
                    raise

    def log_memory_usage(self):
        """记录当前内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        logging.info(f"当前内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")

    def process_data(self, file_path, chunk_size=10000):
        """处理CSV数据并添加到数据库"""
        try:
            logging.info("开始数据处理流程...")
            
            # 获取上次处理进度
            last_progress = self.get_last_progress()
            start_chunk = 0
            start_batch = 0
            total_processed = 0
            
            if last_progress:
                start_chunk, start_batch, total_processed = last_progress
                logging.info(f"找到上次处理进度: 块 {start_chunk}, 批次 {start_batch}, 已处理 {total_processed} 条记录")
            else:
                logging.info("未找到上次处理进度，将从头开始处理")
            
            self.log_memory_usage()
            # 如果没有现有索引，则初始化新索引
            if not hasattr(self, 'index'):
                self.initialize_index(file_path)
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"未找到文件: {file_path}")
            
            # 获取文件大小和总行数
            total_size = os.path.getsize(file_path)
            logging.info(f"处理文件: {file_path}")
            logging.info(f"文件大小: {total_size/1024/1024:.2f}MB")
            
            # 使用新的函数获取行数
            total_rows = get_csv_row_count(file_path)
            if total_rows is None:
                raise ValueError("无法获取CSV文件行数，处理终止")
            elif total_rows == 0:
                raise ValueError("CSV文件为空，没有数据需要处理")
            
            logging.info(f"总行数: {total_rows}")
            
            # 记录开始时间和上次保存的里程碑
            start_time = time.time()
            save_interval = 10000  # 每处理1万条记录保存一次
            last_save_milestone = 0  # 新增：记录上次保存时的处理数量
            
            # 添加区间计时变量
            last_save_time = time.time()
            last_save_processed = 0
            
            logging.info("开始读取CSV文件...")
            # 分块读取CSV
            for chunk_idx, chunk_df in enumerate(pd.read_csv(file_path, dtype=str, chunksize=chunk_size, header=None, names=['term_name', 'category'])):
                logging.info(f"正在处理第 {chunk_idx+1} 块数据...")
                # 跳过已处理的块
                if chunk_idx < start_chunk:
                    continue
                
                chunk_df = chunk_df.fillna("NA")
                logging.info(f"处理第 {chunk_idx+1} 块, 大小: {len(chunk_df)} 条记录")
                
                # 处理每个批次，使用tqdm显示进度条
                batch_ranges = list(range(0, len(chunk_df), self.batch_size))
                with tqdm(batch_ranges, 
                         desc=f"处理第 {chunk_idx+1} 块",
                         unit='批次') as pbar:
                    for batch_idx, start_idx in enumerate(pbar):
                        # 如果是第一个块且有起始批次，跳过已处理的批次
                        if chunk_idx == start_chunk and batch_idx < start_batch:
                            continue
                        
                        end_idx = min(start_idx + self.batch_size, len(chunk_df))
                        batch_df = chunk_df.iloc[start_idx:end_idx]
                        
                        try:
                            self.process_batch(batch_df, file_path)
                            total_processed += len(batch_df)
                            
                            # 更新进度
                            self.update_progress(chunk_idx, batch_idx, total_processed)
                            
                            # 修改后的保存逻辑
                            current_save_milestone = (total_processed // save_interval) * save_interval
                            if current_save_milestone > last_save_milestone:
                                logging.info(f"保存索引...")
                                self.save_index(is_final=False)
                                self.log_memory_usage()
                                current_time = time.time()
                                
                                # 计算区间速度
                                interval_elapsed = current_time - last_save_time
                                interval_processed = total_processed - last_save_processed
                                rate = interval_processed / interval_elapsed if interval_elapsed > 0 else 0
                                
                                progress = (total_processed / total_rows) * 100
                                pbar.set_description(
                                    f"第 {chunk_idx+1} 块 [进度: {progress:.1f}%, 速度: {rate:.1f} 条/秒]"
                                )
                                logging.info(f"已处理 {total_processed} 条记录，完成度: {progress:.1f}%，"
                                           f"当前速度: {rate:.1f} 条/秒")
                                
                                # 更新保存点相关变量
                                last_save_milestone = current_save_milestone
                                last_save_time = current_time
                                last_save_processed = total_processed
                            
                            # API请求后等待
                            time.sleep(self.request_interval)
                            
                        except Exception as e:
                            logging.error(f"处理批次时出错: {e}")
                            continue
            
            # 最终保存
            self.save_index(is_final=True)
            
            # 清理进度记录
            with self.conn:
                self.conn.execute('DELETE FROM processing_progress')
            
            # 报告总体统计
            total_time = time.time() - start_time
            logging.info(f"完成处理 {total_processed} 条记录，用时 {total_time:.2f} 秒")
            logging.info(f"平均处理速度: {total_processed/total_time:.2f} 条/秒")
            
        except Exception as e:
            logging.error(f"处理过程中发生严重错误: {e}")
            raise

    def search(self, query, k=5):
        """搜索相似概念"""
        try:
            # 生成查询向量
            query_vector = self.embed_model._get_text_embedding(query)
            query_vector_np = np.array([query_vector], dtype=np.float32)
            
            # FAISS搜索
            distances, indices = self.index.search(query_vector_np, k)
            
            # 获取结果
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                cursor = self.conn.execute('''
                    SELECT term_name, category, status, created_at
                    FROM financial_terms
                    WHERE rowid = ?
                ''', (int(idx) + 1,))
                result = cursor.fetchone()
                if result:
                    results.append({
                        'term_name': result[0],
                        'category': result[1],
                        'status': result[2],
                        'created_at': result[3],
                        'similarity': float(distance)
                    })
            
            # 测试搜索结果输出
            logging.info(f"\n'{query}' 的搜索结果:")
            for result in results:
                logging.info(f"- {result['term_name']} (类别: {result['category']}, 相似度: {result['similarity']:.4f})")
            
            return results
            
        except Exception as e:
            logging.error(f"搜索过程中出错: {e}")
            raise

    def save(self):
        """保存索引和数据库"""
        try:
            # FAISS索引文件
            index_path = os.path.join(self.db_path, f'{self.db_name}.index')
            faiss.write_index(self.index, index_path)
            logging.info(f"已保存FAISS索引到: {index_path}")
        except Exception as e:
            logging.error(f"保存索引时出错: {e}")
            raise

    def load(self):
        """加载已保存的索引"""
        try:
            index_path = os.path.join(self.db_path, 'financial_vectors.index')
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logging.info(f"已加载FAISS索引: {index_path}")
            else:
                logging.warning(f"未找到索引文件: {index_path}")
        except Exception as e:
            logging.error(f"加载索引时出错: {e}")
            raise

    def initialize_embedding_model(self):
        """初始化embedding模型，增加错误处理"""
        try:
            self.embed_model = ZhipuAIEmbedding(timeout=60)
            # 测试模型是否正常工作
            test_text = "测试文本"
            _ = self.embed_model._get_text_embedding(test_text)
            logging.info("Embedding模型初始化成功")
        except Exception as e:
            logging.error(f"Embedding模型初始化失败: {e}")
            # 清理可能的缓存问题
            importlib.reload(sys.modules.get('zhipu_embedding'))
            raise

    def check_database_integrity(self):
        """检查数据库完整性"""
        try:
            with self.conn:
                # 检查表结构
                self.conn.execute("PRAGMA integrity_check")
                # 检查进度记录的有效性
                cursor = self.conn.execute("""
                    SELECT COUNT(*) FROM processing_progress 
                    WHERE chunk_index < 0 OR batch_start_idx < 0 OR total_processed < 0
                """)
                invalid_records = cursor.fetchone()[0]
                if invalid_records > 0:
                    logging.warning("检测到无效的进度记录，清理进度表")
                    self.conn.execute("DELETE FROM processing_progress")
                    self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"数据库完整性检查失败: {e}")
            return False
        return True

def main():
    try:
        # 初始化数据库，使用修正后的路径
        db = FinancialVectorDB(db_path=DB_PATH, db_name='financial_terms_zhipu')
        
        # 构建完整的数据文件路径
        file_name = "万条金融标准术语.csv"  # 更新为新的文件名
        file_path = os.path.join(DATA_DIR, file_name)
        
        # 检查文件
        logging.info(f"检查文件: {file_path}")
        if not os.path.exists(file_path):
            logging.error(f"文件不存在: {file_path}")
            # 尝试在项目根目录下的 data 目录查找
            alt_data_dir = os.path.join(project_root, 'data')
            alt_file_path = os.path.join(alt_data_dir, file_name)
            if os.path.exists(alt_file_path):
                logging.warning(f"在 {file_path} 未找到文件，但在 {alt_file_path} 找到。将使用后者。")
                file_path = alt_file_path
            else:
                logging.error(f"在 {file_path} 和 {alt_file_path} 均未找到数据文件。请检查文件位置。")
                return
            
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logging.info(f"开始处理文件: {file_path}")
        logging.info(f"文件大小: {file_size_mb:.2f}MB")
        
        # 提示处理时间可能较长
        if file_size_mb > 50:
            logging.info("注意: 文件较大，处理时间可能较长，请耐心等待...")
            logging.info("处理过程中会定期保存进度，可以通过日志文件查看进度")
        
        # 处理数据
        db.process_data(
            file_path,
            chunk_size=10000
        )
        
        # 测试搜索
        test_queries = ["股票", "债券", "期货", "期权", "基金"]
        for query in test_queries:
            results = db.search(query)
            logging.info(f"\n'{query}' 的搜索结果:")
            for result in results:
                logging.info(f"- {result['term_name']} (类别: {result['category']}, 相似度: {result['similarity']:.4f})")
        
    except Exception as e:
        logging.error(f"处理过程中发生错误: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 