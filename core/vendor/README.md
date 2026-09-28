# core/vendor

Private Python wheels the agent core should install at image build time.

Put `cortexa_enterprise_sdk-0.1.0-py3-none-any.whl` here, then `docker compose up -d --build`.
Anything matching `*.whl` in this folder is installed; the image builds without them too.

Only commit a wheel here if the repository is private.
