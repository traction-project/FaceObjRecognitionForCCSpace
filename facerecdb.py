import json
import subprocess
import re
import os
import psycopg2
import time
from flask import Flask
from flask import render_template


def run():
    t_host = "127.0.0.1"
    t_port = "5432"
    t_dbname = "mediavault"
    t_user = "test"
    t_pw = "1234"
    db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
    db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
    db_cursor = db_conn.cursor()

    s = "SELECT title FROM public.multimedia WHERE status='done' AND data_container_id IS NOT NULL"
    # Error trapping
    try:
        # Execute the SQL
        db_cursor.execute(s)
        # Retrieve records from Postgres into a Python List
        list_results = db_cursor.fetchall()
        list_results=[i[0] for i in list_results]
    except psycopg2.Error as e:
        t_message = "Database error: " + e + "/n SQL: " + s
        return render_template("error.html", t_message = t_message)

    # Close the database cursor and connection
    db_cursor.close()
    db_conn.close()

    data = {}

    with open('log.json', "r") as file:
        data = json.load(file)
    
    for i in range(len(list_results)):
        exist = False
        for j in data['log']:
            if j['file'] == str(list_results[i]):
                exist = True

        if exist == False:
            print ('New file: ' + str(list_results[i]))
            data['log'].append({'file': str(list_results[i]),'status': 'new',})

    with open('log.json', 'w') as outfile:
        json.dump(data, outfile)

    #Re-order json with images first
    data = {}
    datanew = {}

    with open('log.json', "r") as file:
        data = json.load(file)

    with open('log.json', "r") as file:    
        datanew = json.load(file)

    for i in range(len(datanew['log'])):
        datanew['log'].pop()      

    for j in data['log']:
        if ((str(j['file'][-3:]) in ('png','PNG','jpg','JPG','peg','PEG'))):
            datanew['log'].append({'file': j['file'],'status':j['status'],})
 
    for j in data['log']:
        if ((str(j['file'][-3:]) not in ('png','PNG','jpg','JPG','peg','PEG'))):
            datanew['log'].append({'file': j['file'],'status':j['status'],})            
        
    with open('log.json', 'w') as outfile:
        json.dump(datanew, outfile)

    data = {}

    with open('log.json', "r") as file:
        data = json.load(file)

    for j in data['log']:
        #Process images: create folder based on post title and add images from aws into it. Index dataset.
        if (j['status'] == 'new') and ((str(j['file'][-3:]) in ('png','PNG','jpg','JPG','peg','PEG'))):
            t_host = "127.0.0.1"
            t_port = "5432"
            t_dbname = "mediavault"
            t_user = "test"
            t_pw = "1234"
            db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
            db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
            db_cursor = db_conn.cursor()
            # SQL to get records from Postgres - CHECK IF DATA CONTAINER EXISTS AND POST_ID DOES NOT WHICH INDICATE DELETED POST CHANGE STATUS TO ERROR - SECOND SELECT SAME CURSOR. IF DATA CONTAINER DOES NOT EXIST CONTINUE STATUS AS NEW
            s = ("SELECT title FROM public.posts WHERE id = (SELECT post_id FROM public.data_container WHERE id = (SELECT data_container_id FROM public.multimedia WHERE title = '"+j['file']+"'))")
            # Error trapping
            try:
                # Execute the SQL
                db_cursor.execute(s)
                # Retrieve records from Postgres into a Python List
                list_results = db_cursor.fetchall()
                list_results=[i[0] for i in list_results]
            except psycopg2.Error as e:
                t_message = "Database error: " + e + "/n SQL: " + s
                return render_template("error.html", t_message = t_message)

            # Close the database cursor and connection
            db_cursor.close()
            db_conn.close()

            for i in range(len(list_results)):
                print ('Face/object: ' + list_results[i])

            if (len(list_results) != 0):
                newpath = ('C:/Users/ads_s/OneDrive/Documentos/facetool-master/facetool-master/media/images/' + str(list_results[i]).capitalize())
                
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                
                if not os.path.exists(newpath+'/'+j['file']):
                    print (newpath+'/'+j['file'])
                    subprocess.call ('aws s3 cp s3://traction-mediavault/' + j['file']+ ' "'+newpath+'"')
                subprocess.call ('python3 facetool.py index media/images media/index')
                j['status']="done"

        with open('log.json', 'w') as outfile:
            json.dump(data, outfile)

        #Process videos: Download videos from aws, detect tags and add to database
        if (j['status'] == 'new') and ((str(j['file'][-3:]) in ('mp4','MP4','mov','MOV','m4v','M4V'))):

            if not os.path.exists('C:/Users/ads_s/OneDrive/Documentos/facetool-master/facetool-master/media/'+j['file']):
                subprocess.call ('aws s3 cp s3://traction-mediavault/' + j['file']+ ' C:/Users/ads_s/OneDrive/Documentos/facetool-master/facetool-master/media')
            txtfile = open ("facesrec.txt", "w")
            subprocess.call ('python3 facetool.py recognize media/'+j['file']+ ' media/index', stdout=txtfile)

            pattern = '] 'r'\w+'' 0'

            tags = {}
            tags = set()

            for i, line in enumerate(open('facesrec.txt')):
                for match in re.finditer(pattern, line):
                    tags.add(match.group()[2:-2])
        
            tags.remove('None')
            
            tagslist = list(tags)

            for i in range(len(tagslist)):

                t_host = "127.0.0.1"
                t_port = "5432"
                t_dbname = "mediavault"
                t_user = "test"
                t_pw = "1234"
                db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
                db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
                db_cursor = db_conn.cursor()
                # SQL to get records from Postgres

                print ("tag:"+tagslist[i])

                s = ("INSERT INTO public.tags (id, created_at, updated_at, tag_name) VALUES (uuid_generate_v4(), now(), now(), '"+tagslist[i]+"') ON CONFLICT DO NOTHING")
                # Error trapping
                try:
                    # Execute the SQL
                    db_cursor.execute(s)
                    db_conn.commit()
                    # Retrieve records from Postgres into a Python List

                except psycopg2.Error as e:
                    t_message = "Database error: " + e + "/n SQL: " + s
                    return render_template("error.html", t_message = t_message)
                db_cursor.close()
                db_conn.close()

            for i in range(len(tagslist)):

                t_host = "127.0.0.1"
                t_port = "5432"
                t_dbname = "mediavault"
                t_user = "test"
                t_pw = "1234"
                db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
                db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
                db_cursor = db_conn.cursor()
                # SQL to get records from Postgres SELECT POST ID SEPARATELY, IF NULL, SAVE STATUS AS ERROR - test m4v

                s = ("INSERT INTO public.tag_references (created_at, updated_at, tag_id, post_id) VALUES (now(), now(), (SELECT id FROM public.tags WHERE tag_name = '"+tagslist[i]+"'), (SELECT post_id FROM public.data_container WHERE id = (SELECT data_container_id FROM public.multimedia WHERE title = '"+j['file']+"'))) ON CONFLICT DO NOTHING")
                # Error trapping
                try:
                    # Execute the SQL
                    db_cursor.execute(s)
                    db_conn.commit()
                    
                except psycopg2.Error as e:
                    t_message = "Database error: " + e + "/n SQL: " + s
                    return render_template("error.html", t_message = t_message)
                db_cursor.close()
                db_conn.close()
            
            j['status']="done"

        with open('log.json', 'w') as outfile:
            json.dump(data, outfile) 
    time.sleep(5)
    
             
if __name__ == '__main__':
    from sys import argv
    print ('CC Space Face Recognition running...')
    while True:
        run()