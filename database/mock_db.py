import sqlite3
import threading
from pathlib import Path
import psycopg2
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# DB_PATH = PROJECT_ROOT / 'db' / 'employees.db'

PG_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'dbname': os.getenv('POSTGRES_DB', 'hr_agent'),
    'user': os.getenv('POSTGRES_USER', 'hr'),
    'password': os.getenv('POSTGRES_PASSWORD', 'hr_password'),
}
_query_lock = threading.Lock()

def get_connection(config: dict = None) -> psycopg2.extensions.connection:
    """
    业务运行时连接函数，仅连接并开启外健
    """
    cfg = config or PG_CONFIG
    try:
        conn = psycopg2.connect(**cfg)
        conn.autocommit = True          # 查询为主，开启自动提交避免长事务
        return conn
    except psycopg2.OperationalError as e:
        raise ConnectionError(f'错误，无法连接 PostgreSQL （{cfg["host"]}:{cfg["port"]}/{cfg['dbname']}）:{e}\n'
                              f'请先执行 docker compose up -d 拉起数据库并确认 .env 中的配置')

    # if not db_path.exists():
    #     raise FileNotFoundError(
    #         f'[错误] 数据库文件未找到：{db_path} \n'
    #         f'请先在终端手动运行一遍初始化脚本： python database/mock_db.py \n'
    #     )
    #
    # conn = sqlite3.connect(str(db_path), check_same_thread=False)
    # conn.execute('PRAGMA foreign_keys = ON')
    # return conn

def init_db(config:dict = None) -> psycopg2.extensions.connection:
    """
    数据库初始化语数据落盘（仅手动单次运行）
    """
    # db_path.parent.mkdir(parents=True, exist_ok=True)

    # 连接数据库
    conn = get_connection(config=config)
    cursor = conn.cursor()  # 游标

    # conn = sqlite3.connect(str(db_path), check_same_thread=False)
    # conn.execute('PRAGMA foreign_keys = ON')
    # cursor = conn.cursor()      # 游标

    # 创建员工表
    cursor.execute("""
    create table if not exists employees (
        uid text primary key,               -- 员工唯一标识
        name text not null,                 -- 员工姓名
        rank text not null,                 -- 职级（P3、P4）
        location text not null,             -- 工作地点（城市名称）
        seniority integer not null,         -- 入职年限（年）
        base_salary integer not null        -- 基本工资（元）
    )
    """)

    # 创建假期表
    cursor.execute('''
    create table if not exists leave_balances (
        uid text primary key,                       -- 员工唯一标识(外健关联 employees.id)
        annual_leave_remaining integer not null,    -- 剩余年假天数
        sick_leave_remaining integer not null,      -- 剩余病假天数
        foreign key (uid) references employees (uid)
    )
    ''')

    # 清空旧数据（确保幂等性）先删除子表再删除主表
    cursor.execute('''delete from leave_balances''')
    cursor.execute('''delete from employees''')

    # 注入一些数据
    test_employees = [
        ('1001', '张三', 'P5', '北京', 2, 18000),
        ('1002', '李四', 'P4', '成都', 4, 9000),
        ('1003', '王五', 'P7', '上海', 5, 35000),
        ('1004', '赵六', 'P3', '深圳', 0, 7500),
    ]

    test_balances = [
        ('1001', 6, 10),
        ('1002', 7, 12),
        ('1003', 14, 15),
        ('1004', 2, 5),
    ]

    cursor.executemany("insert into employees values (%s, %s, %s, %s, %s, %s)", test_employees)
    cursor.executemany("insert into leave_balances values (%s, %s, %s)", test_balances)

    conn.commit()
    cursor.close()

    print('「成功」实体数据库已成功落盘')
    print(f'数据库路径：{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG["dbname"]}')
    return conn

def query_db(conn:psycopg2.extensions.connection, sql:str, params:tuple = ()):
    """通用查询函数"""
    with _query_lock:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        return rows
    # cursor = conn.cursor()
    # cursor.execute(sql, params)
    # columns = [col[0] for col in cursor.description]    # 获取元数据，col[0] 包含表的列名
    # return [dict(zip(columns, row)) for row in cursor.fetchall()]
"""
cursor.fetchall() : 获取所有行
zip(columns, row)：将列名和值配对
dict：将配对转换成字典
"""
def close_db(conn:psycopg2.extensions.connection):
    """安全关闭数据库"""
    if conn:
        conn.close()
        print('数据库连接已安全关闭。')

# 手动运行初始化
if __name__ == '__main__':
    print('正在执行数据库手动初始化操作')
    standalone_conn = init_db()
    close_db(standalone_conn)