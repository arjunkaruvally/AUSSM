build_cuda: extension-cpp/setup.py
	if [ "${BUILD_ENVIRONMENT}" = "debug" ]; then USE_CUDA=1 DEBUG=1 python -m pip install --no-build-isolation -v -e extension-cpp/.; fi
	if [ "${BUILD_ENVIRONMENT}" = "deploy" ]; then USE_CUDA=1 DEBUG=0 TORCH_CUDA_ARCH_LIST="Turing;Ampere+Tegra;Ampere;Ada;Hopper;Volta" python -m pip install --no-build-isolation -v extension-cpp/.; fi

build: build_cuda ai-modules/setup.py
	python -m pip install -c constraints.txt -r requirements.txt;
	if [ "${BUILD_ENVIRONMENT}" = "deploy" ]; then \
		echo "WARNING: environment with pytorch==2.4.0 and a compatible cuda has to be made before "; \
		TORCH_CUDA_ARCH_LIST="Turing;Ampere+Tegra;Ampere;Ada;Hopper;Volta"  python -m pip install -c constraints.txt --no-build-isolation -v mamba-ssm; \
		python -m pip install -c constraints.txt --no-build-isolation -v ai-modules/.; \
	fi
	if [ "${BUILD_ENVIRONMENT}" = "debug" ]; then \
		echo "WARNING: environment with pytorch==2.4.0 and a compatible cuda has to be made before "; \
		python -m pip install -c constraints.txt --no-build-isolation -v mamba-ssm; \
		python -m pip install -c constraints.txt --no-build-isolation -v -e ai-modules/.; \
	fi

test_cuda:
	python -m pytest -s extension-cpp/.

clean:
	python -m pip uninstall mamba-ssm
	python -m pip uninstall extension_cpp
	python -m pip uninstall wavesAI
	python -m pip uninstall -y -r requirements.txt


cuda_benchmark:
    python extension-cpp/test/scaling_benchmark.py


algorithmic_tasks:
    ./experiments/all/algorithmic.sh


ts_classification:
    ./experiments/all/ts_classification.sh

ts_regression:
    ./experiments/all/ts_regression.sh
