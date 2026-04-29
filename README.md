# Rag PDF (Projeto de Estudos)

Este repositório é um projeto de estudo que implementa uma pipeline de leitura de PDFs com suporte a retrieval-augmented generation (RAG) via uma API simples e uma interface minimalista baseada em front-end. O objetivo é oferecer um fluxo didático para aprender como transformar PDFs em respostas informativas.

## Estrutura do repositório (referência das pastas existentes)
- notebooks: notebooks de experimentação
- data: dados de exemplo e arquivos de índice/base de PDFs
- src: código fonte principal (API, front-end, injecção de dados, etc.)
- .exemple-env, .gitignore, .dockerignore, Dockerfile, docker-compose.yml, pyproject.toml, requirements.txt: configuração do ambiente e containers
- README.md: este arquivo

Obs.: os nomes das pastas são usados exatamente como aparecem no repositório. Este projeto é voltado para estudos, então mantenha o setup simples para facilitar a reprodução.

## Installing and Running (Getting Started rápido)
- Pré-requisitos: Python 3.x (conforme seu ambiente local) e Docker (opcional, para o pipeline com containers)
- Opção 1: usar Docker (recomendado para isolamento):
  - docker-compose build
  - docker-compose up -d
- Opção 2: executar localmente (sem Docker):
  - python -m venv venv
  - source venv/bin/activate  # Linux/macOS
  - pip install -r requirements.txt
  - python -m src.api  # ajuste conforme o entrypoint disponível

## Como usar (fluxo básico)
- Ingestão de PDF: coloque um arquivo PDF em data/pdfs/ e execute os utilitários de ingestão (se existirem no projeto).
- Consulta: através da API, envie uma pergunta relacionada ao conteúdo do PDF; exemplo de endpoint esperado dependerá da implementação atual.
- Observação: como este é um projeto de estudo, alguns caminhos e scripts podem variar entre as execuções.

## Testes
- Executar pytest: `pytest -q`
- Por ser um projeto de estudo, os testes podem estar esboçados; adapte ou complemente conforme necessário.

## Conclusões principais (simples)
- O objetivo central é demonstrar o fluxo de transformar PDFs em respostas usando um pipeline RAG.
- Estrutura clara do projeto facilita a exploração e aprendizado (Ingestão -> Indexação -> Recuperação -> Geração de Resposta).
- A configuração minimalista facilita reproduzir o ambiente local com ou sem Docker.

## Contribuição
- Este é um projeto de estudo. Contribuições são bem-vindas para aprendizado e melhoria do pipeline.
- Regras básicas: crie um branch para a feature, mantenha commits curtos e com mensagens descritivas, abra issues/PRs com o objetivo da mudança.

## Licença
- MIT (ou licenca aplicável)

## Contato / Suporte
- Dúvidas: abra uma issue no repositório.
