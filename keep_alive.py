import threading
import time
import requests
import subprocess
import os
from flask import Flask

# Flask ping server (NE na glavnom portu!)
app = Flask(__name__)

@app.route("/")
def home():
    return "alive"

def ping():
    url = "https://rag-document-assistant-gt.onrender.com"
    while True:
        try:
            print("Pinging backend...")
            requests.get(url, timeout=10)
        except Exception as e:
            print("Ping error:", e)
        time.sleep(300)

def start_ping():
    t = threading.Thread(target=ping)
    t.daemon = True
    t.start()

def start_streamlit():
    port = os.environ.get("PORT", "10000")
    print("Starting Streamlit on port", port)

    cmd = [
        "streamlit", "run", "app/main.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0"
    ]
    subprocess.Popen(cmd)

if __name__ == "__main__":
    start_ping()
    start_streamlit()
    # Flask IDE na drugom portu da ne smeta Streamlitu
    app.run(host="0.0.0.0", port=8080)
