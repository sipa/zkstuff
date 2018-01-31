SAGE := /opt/SageMath/sage

jubjub-%.program: jubjub.sage
	$(SAGE) $< $* >$@

jubjub-%.circuit: jubjub-%.program circuitify.py
	./circuitify.py <$< >$@

jubjub-circuits: jubjub-3.circuit jubjub-6.circuit jubjub-12.circuit jubjub-24.circuit jubjub-48.circuit jubjub-96.circuit jubjub-192.circuit jubjub-384.circuit jubjub-768.circuit jubjub-1536.circuit jubjub-3072.circuit
