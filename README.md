# o9-platform-wiki-retrieval

## Tools
<div align="center">

[![Python](https://img.shields.io/badge/python-3.11.7-34d058?logo=python)](https://www.python.org/downloads/release/python-3117/)
[![FastAPI](https://img.shields.io/badge/fastapi-v0.109.2-34d058?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/%F0%9F%A6%9C%EF%B8%8Flangchain-v0.1.7-34d058)](https://www.langchain.com/)
[![Uvicorn](https://img.shields.io/badge/uvicorn-0.27.1-34d058?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPCEtLSBSZXBsYWNlIHRoZSBjb250ZW50cyBvZiB0aGlzIGVkaXRvciB3aXRoIHlvdXIgU1ZHIGNvZGUgLS0%2BCgo8c3ZnIHJvbGU9ImltZyIgdmlld0JveD0iMCAwIDI0IDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxwYXRoIGQ9Ik0xMiA4YTMgMyAwIDAgMCAzLTMgMyAzIDAgMCAwLTMtMyAzIDMgMCAwIDAtMyAzIDMgMyAwIDAgMCAzIDNtMCAzLjU0QzkuNjQgOS4zNSA2LjUgOCAzIDh2MTFjMy41IDAgNi42NCAxLjM1IDkgMy41NCAyLjM2LTIuMTkgNS41LTMuNTQgOS0zLjU0VjhjLTMuNSAwLTYuNjQgMS4zNS05IDMuNTRaIj48L3BhdGg%2BCjwvc3ZnPg%3D%3D&logoColor=white)](https://www.uvicorn.org/)<br>
[![OpenAI](https://img.shields.io/badge/openai-412991?style=for-the-badge&logo=openai)](https://openai.com/)
[![Qdrant](https://img.shields.io/badge/qdrant-d02653?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBzdHlsZT0ic2hhcGUtcmVuZGVyaW5nOmdlb21ldHJpY1ByZWNpc2lvbjt0ZXh0LXJlbmRlcmluZzpnZW9tZXRyaWNQcmVjaXNpb247aW1hZ2UtcmVuZGVyaW5nOm9wdGltaXplUXVhbGl0eTtmaWxsLXJ1bGU6ZXZlbm9kZDtjbGlwLXJ1bGU6ZXZlbm9kZCI+PHBhdGggZmlsbD0iIzc5OGJiZSIgZD0iTTkzLjUtLjVoMTJhMzQxMS45MzUgMzQxMS45MzUgMCAwIDAgODIgNDggMjM1MC43MyAyMzUwLjczIDAgMCAxIDAgMTM3Yy4xNjctNDUuMzM1IDAtOTAuNjY4LS41LTEzNmExMzE1LjU5OCAxMzE1LjU5OCAwIDAgMS0yOS41IDE3IDE2MjAuNjExIDE2MjAuNjExIDAgMCAwLTU3LjUtMzMgMzI1OS4yODUgMzI1OS4yODUgMCAwIDAtNTguNSAzMyA0MTcuMDk2IDQxNy4wOTYgMCAwIDAtMjQtMTMuNWMtMi4xOTItMS4wNjctMy44NTgtMi41NjctNS00LjVsODEtNDhaIiBzdHlsZT0ib3BhY2l0eTouOTk5Ii8+PHBhdGggZmlsbD0iI2RiMjU0YyIgZD0iTTE1Ny41IDY1LjV2MTM0aC00YTYxOC4wMyA2MTguMDMgMCAwIDAtMjYtMTQgMzA3LjE2MiAzMDcuMTYyIDAgMCAwLTEtMzUgNTUzLjczNyA1NTMuNzM3IDAgMCAxLTI3IDE2Yy0uNjY3IDAtMS0uMzMzLTEtMSAuOTk1LTExLjE1NCAxLjMyOC0yMi40ODggMS0zNGEyMzEuOTA3IDIzMS45MDcgMCAwIDAgMjcuNS0xNmMuNS0xMC42NjEuNjY3LTIxLjMyOC41LTMyYTI4OS4wMjUgMjg5LjAyNSAwIDAgMC0yOC0xNyAyOTEuNzk1IDI5MS43OTUgMCAwIDAtMjggMTdjLS4xNjYgMTAuNjcyIDAgMjEuMzM5LjUgMzJhMTU4LjM2IDE1OC4zNiAwIDAgMSAxMC41IDdjLjMyOCAxMS4xNzktLjAwNSAyMi4xNzktMSAzM2ExMTM0Ljc0NyAxMTM0Ljc0NyAwIDAgMS0zOS0yMmMtMS0yMi42NTctMS4zMzMtNDUuMzIzLTEtNjhhMzI1OS4yODUgMzI1OS4yODUgMCAwIDEgNTguNS0zMyAxNjIwLjYxMSAxNjIwLjYxMSAwIDAgMSA1Ny41IDMzWiIgc3R5bGU9Im9wYWNpdHk6MSIvPjxwYXRoIGZpbGw9IiNiMmJmZTciIGQ9Ik0xMi41IDQ3LjVjMS4xNDIgMS45MzMgMi44MDggMy40MzMgNSA0LjVhNDE3LjA5NiA0MTcuMDk2IDAgMCAxIDI0IDEzLjVjLS4zMzMgMjIuNjc3IDAgNDUuMzQzIDEgNjhhMTEzNC43NDcgMTEzNC43NDcgMCAwIDAgMzkgMjIgMjExLjg3MiAyMTEuODcyIDAgMCAwIDE3IDEwYzAgLjY2Ny4zMzMgMSAxIDF2MzNoLTRhMjg2OTYuNjkgMjg2OTYuNjkgMCAwIDEtODMtNDggMTM1NS42NzYgMTM1NS42NzYgMCAwIDEgMC0xMDRaIiBzdHlsZT0ib3BhY2l0eToxIi8+PHBhdGggZmlsbD0iIzI3Mzg2YyIgZD0iTTE4Ny41IDE4NC41YTQ1My40ODUgNDUzLjQ4NSAwIDAgMC0yNSAxNWgtNXYtMTM0YTEzMTUuNTk4IDEzMTUuNTk4IDAgMCAwIDI5LjUtMTdjLjUgNDUuMzMyLjY2NyA5MC42NjUuNSAxMzZaIiBzdHlsZT0ib3BhY2l0eToxIi8+PHBhdGggZmlsbD0iIzc2ODdiYyIgZD0ibTEyNy41IDgzLjUtMjggMTUtMjgtMTVhMjkxLjc5NSAyOTEuNzk1IDAgMCAxIDI4LTE3IDI4OS4wMjUgMjg5LjAyNSAwIDAgMSAyOCAxN1oiIHN0eWxlPSJvcGFjaXR5OjEiLz48cGF0aCBmaWxsPSIjYjJiZWU3IiBkPSJtNzEuNSA4My41IDI4IDE1djMzYTIwNy40MjggMjA3LjQyOCAwIDAgMC0xNy05IDE1OC4zNiAxNTguMzYgMCAwIDAtMTAuNS03IDUxMi40NjMgNTEyLjQ2MyAwIDAgMS0uNS0zMloiIHN0eWxlPSJvcGFjaXR5OjEiLz48cGF0aCBmaWxsPSIjMjYzOTZkIiBkPSJNMTI3LjUgODMuNWMuMTY3IDEwLjY3MiAwIDIxLjMzOS0uNSAzMmEyMzEuOTA3IDIzMS45MDcgMCAwIDEtMjcuNSAxNnYtMzNsMjgtMTVaIiBzdHlsZT0ib3BhY2l0eToxIi8+PHBhdGggZmlsbD0iI2ZjMzM2MyIgZD0iTTgyLjUgMTIyLjVhMjA3LjQyOCAyMDcuNDI4IDAgMCAxIDE3IDljLjMyOCAxMS41MTItLjAwNSAyMi44NDYtMSAzNGEyMTEuODcyIDIxMS44NzIgMCAwIDEtMTctMTBjLjk5NS0xMC44MjEgMS4zMjgtMjEuODIxIDEtMzNaIiBzdHlsZT0ib3BhY2l0eToxIi8+PHBhdGggZmlsbD0iIzI5M2I2ZSIgZD0iTTEyNy41IDE4NS41YTM4Ni4wMzMgMzg2LjAzMyAwIDAgMC0yMyAxNGgtNXYtMzNhNTUzLjczNyA1NTMuNzM3IDAgMCAwIDI3LTE2IDMwNy4xNjIgMzA3LjE2MiAwIDAgMSAxIDM1WiIgc3R5bGU9Im9wYWNpdHk6Ljk5OSIvPjwvc3ZnPg==)](https://qdrant.tech/)<br>
[![Pipenv](https://img.shields.io/badge/pipenv-v2023.12.1-3776ab)](https://pipenv.pypa.io/en/latest/)
</div>

## Installation

_Install_ `pipenv` to manage the project virtual environment. Don't forget to check.
```bash
$ pip install pipenv --user
$ pipenv --version
pipenv, version 2023.12.1
```

_Make_ a virtual environment and install packages in your local project directory. The name is automatically created based on the project folder name.
```bash
$ cd YOUR-PRJ-PATH
$ pipenv --python 3.11.7 sync
```

_Run_ a streamit app
```
$ streamlit run src/app/app.py
```
