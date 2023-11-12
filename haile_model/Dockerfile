FROM python:3.9
ARG CI_JOB_TOKEN
ARG CI_COMMIT_TAG

WORKDIR /app/alchemy

RUN pip install poetry
COPY alchemy/pyproject.toml alchemy/poetry.lock .

ENV POETRY_HTTP_BASIC_OAI_USERNAME gitlab-ci-token
ENV POETRY_HTTP_BASIC_OAI_PASSWORD $CI_JOB_TOKEN

RUN poetry config virtualenvs.create false
RUN poetry install --no-root --without=dev

COPY alchemy/ .
RUN echo $CI_COMMIT_TAG > tag

ENTRYPOINT ["/bin/bash"]

