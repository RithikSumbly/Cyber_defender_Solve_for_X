.PHONY: setup train calibrate serve demo eval verify dashboard walkthrough kill clean

PY := python3

setup:
	$(PY) -m pip install -q -r requirements.txt
	$(MAKE) train
	$(MAKE) calibrate
	@echo "[setup] done: model trained and detector calibrated."

train:
	$(PY) -m sentry.model.train_victim

calibrate:
	@( $(PY) -m sentry.api.server & echo $$! > .server.pid ) ; \
	sleep 3 ; \
	$(PY) -m sentry.detect.calibrate ; \
	kill `cat .server.pid` 2>/dev/null; rm -f .server.pid

serve:
	$(PY) -m sentry.api.server

demo:
	$(PY) scripts/demo.py

eval:
	@( $(PY) -m sentry.api.server & echo $$! > .server.pid ) ; \
	sleep 3 ; \
	$(PY) -m sentry.eval.run_eval ; \
	kill `cat .server.pid` 2>/dev/null; rm -f .server.pid

verify:
	$(PY) scripts/verify.py

dashboard:
	@echo "[dashboard] needs \`make serve\` and \`python -m sentry.detect.detector\` running in other panes"
	$(PY) -m sentry.dashboard.server

walkthrough:
	$(PY) -m sentry.demo.server

kill:
	-pkill -f "sentry.api.server"
	-pkill -f "sentry.detect.detector"
	-pkill -f "sentry.dashboard.server"
	-pkill -f "sentry.demo.server"
	rm -f .server.pid

clean:
	rm -f results/traffic_log.jsonl results/alerts.jsonl
