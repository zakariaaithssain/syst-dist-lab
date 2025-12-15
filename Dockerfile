FROM python:3.12.12
WORKDIR /DIST_SYS
# 3. Install curl (needed to install UV) and other essentials
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
# 4. Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# 5. Add UV to PATH
ENV PATH="/root/.local/bin:${PATH}"

COPY . . 
RUN uv venv .venv
RUN uv pip install -r requirements.txt --python .venv/bin/python

EXPOSE 3000

CMD ["python", "-m", "app.main"]

