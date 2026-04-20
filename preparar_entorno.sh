# Preparar Entorno
deactivate
rm -rf venv

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip # opcional pero recomendado
pip install --no-cache-dir -r requirements.txt

# Ejecutar setup.py
python setup.py build_ext --inplace
