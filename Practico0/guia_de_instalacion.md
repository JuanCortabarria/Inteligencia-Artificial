# Practicos IA

## Guía de instalación

### Instalar Python 3.10

https://www.python.org/downloads/

### Instalar Poetry

#### Linux / MacOS

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### Windows

```Powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Configurar poetry para que instale el .venv junto con el proyecto

```bash
poetry config virtualenvs.in-project true
```
