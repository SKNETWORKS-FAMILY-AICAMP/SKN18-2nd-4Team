import pymysql

def connect_db():
    con = pymysql.connect(host='localhost', user='root', password='root1234',
                        db='car', charset='utf8') 
    return con 
