# Image Lambda Python 3.12 — inférence ONNX, sans PyTorch.
# Build (à la racine du repo, après `valeo-qc export`) :
#   docker build -f deployment/Dockerfile -t valeo-qc-lambda .
# Test local (Runtime Interface Emulator, port 8080) :
#   docker run --rm -p 9000:8080 -e VALEO_MODELS_DIR=/var/task/models valeo-qc-lambda
#   curl -s -XPOST http://localhost:9000/2015-03-31/functions/function/invocations -d "{\"image_b64\":\"...\"}"

FROM public.ecr.aws/lambda/python:3.12

ENV VALEO_MODELS_DIR=/var/task/models

COPY deployment/requirements-lambda.txt ${LAMBDA_TASK_ROOT}/requirements-lambda.txt
# trusted-host : réseaux d'entreprise (inspection TLS) cassent le CA de l'image Lambda
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

COPY src/valeo_qc ${LAMBDA_TASK_ROOT}/valeo_qc
COPY deployment/lambda_handler.py ${LAMBDA_TASK_ROOT}/lambda_handler.py
COPY models/threshold.json models/onnx-manifest.json ${LAMBDA_TASK_ROOT}/models/
COPY models/classifier.onnx models/padim-backbone.onnx ${LAMBDA_TASK_ROOT}/models/

CMD ["lambda_handler.handler"]
