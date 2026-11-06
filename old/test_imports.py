import sys
import os

os.path.dirname(__file__)
#folder_to_add = os.path.abspath( os.path.join(os.path.dirname(__file__), r'Lib\site-packages\PyOpenColorIO\bin'))
sys.path.append(os.path.dirname(os.path.abspath( os.path.join(os.path.dirname(__file__), r'Lib\site-packages\PyOpenColorIO\bin\\'))))
sys.path.append(os.path.dirname(os.path.abspath( os.path.join(os.path.dirname(__file__), r'Lib\site-packages\PyOpenColorIO\\'))))

# test_imports.py
print("Iniciando teste de importação...")

try:
    print("Tentando importar 'av' (PyAV)...")
    import av
    print("'av' importado com sucesso!")

    print("\nTentando importar 'PyOpenColorIO'...")
    import PyOpenColorIO as OCIO
    print("'PyOpenColorIO' importado com sucesso!")

    print("\nTodos os imports críticos funcionaram.")

except Exception as e:
    print(f"Um erro ocorreu durante a importação: {e}")

input("\nTeste concluído. Pressione Enter para sair.")
