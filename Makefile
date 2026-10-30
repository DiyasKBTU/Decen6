PYTHON ?= python
DATA ?= ../grand-support-copilot-handout-2026-10-27/handout
PORT ?= 8000
IN ?= $(DATA)/tickets_valid.jsonl
OUT ?= out/pred.jsonl

export COPILOT_DATA_DIR := $(DATA)

.PHONY: setup artifacts index dedup train templates serve queue predict test eval eval-retrieval eval-dry replay \
	docker-build docker-predict

# ---------- 1. Подготовка (один раз) ----------

setup:
	$(PYTHON) -m pip install -r requirements.txt

# Всё, что строится из train, кладётся в artifacts/ (вне git, ~10 мин):
# база прошлых ответов, история для дублей, классификатор, кластеры для новых шаблонов
artifacts: index dedup train templates

index:
	$(PYTHON) scripts/build_history_index.py

dedup:
	$(PYTHON) scripts/build_dedup_history.py

train:
	$(PYTHON) scripts/train_classifier.py

# Массовые прогоны по train: похожие ответы (RAG-примеры) им не нужны и только замедляют
templates:
	COPILOT_LLM=0 COPILOT_DEDUP=0 COPILOT_RAG_EXAMPLES=0 $(PYTHON) scripts/suggest_templates.py --input $(DATA)/tickets_train.jsonl.gz

# ---------- 2. Работа сервиса ----------

# make predict IN=tickets.jsonl OUT=pred.jsonl
# LLM включается сам, если задан ANTHROPIC_API_KEY (в окружении или .env); COPILOT_LLM=0 — только шаблоны
predict:
	$(PYTHON) -m copilot.predict --in $(IN) --out $(OUT)

serve:
	$(PYTHON) -m uvicorn copilot.api:app --host 0.0.0.0 --port $(PORT)

# Демо-очередь для интерфейса админа: 300 обращений valid с ответами админов для сравнения
queue:
	$(PYTHON) scripts/load_queue.py --input $(DATA)/tickets_valid.jsonl --limit 300 --with-reference

# ---------- 3. Качество ----------

test:
	$(PYTHON) -m pytest -q

# Метрики на valid -> out/eval_current.json и out/report_current.md
eval:
	$(PYTHON) scripts/eval_valid.py --input $(DATA)/tickets_valid.jsonl --stage current

# Качество поиска по правилам (RAG) -> out/eval_retrieval.json
eval-retrieval:
	COPILOT_DEDUP=0 $(PYTHON) scripts/eval_retrieval.py --input $(DATA)/tickets_valid.jsonl

# Ветка LLM без API: фейковая модель возвращает шаблон
eval-dry:
	$(PYTHON) scripts/eval_valid.py --input $(DATA)/tickets_valid.jsonl --stage dry --fake-llm echo

# Прогон истории: время до ответа на жалобы train с нашей очередью -> out/replay.md
replay:
	COPILOT_LLM=0 COPILOT_RAG_EXAMPLES=0 $(PYTHON) scripts/replay_history.py --input $(DATA)/tickets_train.jsonl.gz

# ---------- 4. Docker (данные и artifacts монтируются снаружи, в образ не попадают) ----------

docker-build:
	docker build -t support-copilot .

docker-predict:
	docker run --rm -e ANTHROPIC_API_KEY -v "$(abspath $(DATA))":/data -v "$(abspath artifacts)":/app/artifacts \
		-v "$(abspath $(dir $(OUT)))":/out \
		support-copilot python -m copilot.predict --in /data/$(notdir $(IN)) --out /out/$(notdir $(OUT))
