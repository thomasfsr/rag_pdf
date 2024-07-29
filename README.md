# RAG application with Streamlit Interface  
  
## Project Structure:  
notebooks  
  └── notebook.ipynb          # Jupyter notebooks for experimentation and prototyping  
data  
  ├── .lancedb                # LanceDB files for vector storage  
  └── file_loaded.pdf         # Exemple of a PDF loaded  
src  
  ├── __init__.py             # Initialize the src module  
  ├── api.py                  # FastAPI server   
  ├── main_frontend.py        # Main frontend application logic   
  ├── pdf_frontend.py         # Frontend for PDF loading    
  ├── query_frontend.py       # Frontend for chatting  
  ├── query_data_simple.py    # Simple query processing  
  ├── query_data_branches.py  # Query processing with branches  
  ├── query_data_with_tools.py # Query processing with external tools  
  └── vector_db.py            # Vector database integration  
.exemple-env                  # Example environment configuration file  
.gitignore                    # Git ignore rules  
.dockerignore                 # Docker ignore rules  
Dockerfile                    # Dockerfile for containerized deployment  
docker-compose.yml            # Docker Compose configuration  
python-version                # Python version specification  
poetry.lock                   # Poetry lock file for dependencies  
pyproject.toml                # Project configuration file  
README.md                     # Project documentation  
  
## Install and Run:  
To run the application locally just clone the repo:  
```bash
git clone https://github.com/thomasfsr/rag_pdf  
```
Once inside the project's directory, just build the docker-compose:  
```bash
docker-compose build  
```
Finally, run the docker-compose and it is up and running:  
```bash
docker-compose up -d
```
## Tests  
You can test the API using the built-in documentation FastAPI provides using the following link in the browser:  
```link
https://fastapi:8000/docs
```