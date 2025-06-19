# main.py
import subprocess
import sys
import os

def run_script(script_name):
    
    print(f"\n--- Executando {script_name} ---")
    try:
        result = subprocess.run([sys.executable, script_name], check=False, capture_output=False, text=False)
        if result.returncode != 0:
            print(f"Erro ao executar {script_name}. Código de saída: {result.returncode}")
    except FileNotFoundError:
        print(f"Erro: O script '{script_name}' não foi encontrado. Verifique o caminho.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao executar {script_name}: {e}")
    print(f"--- Fim da execução de {script_name} ---\n")

def display_menu():
    print("="*30)
    print("Menu Principal do Projeto Gatos IA")
    print("="*30)
    print("1. Treinar Modelo (treino_gatos.py)")
    print("2. Avaliar Modelo (avaliar_modelo.py)")
    print("3. Visualizar Aumento de Dados (visualizar_aumento.py)")
    print("4. Testar Gatos Desconhecidos (testar_gatos_desconhecidos.py)") 
    print("0. Sair")
    print("="*30)

def main():
    
    while True:
        display_menu()
        choice = input("Digite sua escolha (0-3): ").strip()

        if choice == '1':
            run_script('treino_gatos.py')
        elif choice == '2':
            # Antes de avaliar, verifique se o modelo foi treinado e salvo
            model_path = os.path.join("modelos", "best_gatos_classifier.pth")
            if not os.path.exists(model_path):
                print("\nAVISO: O modelo ainda não foi treinado ou salvo. Por favor, treine o modelo primeiro (Opção 1).")
                _ = input("Pressione Enter para continuar...") # Pausa para o usuário ler o aviso
            run_script('avaliar_modelo.py')
        elif choice == '3':
            run_script('visualizar_aumento.py')
        elif choice == '4':
            model_path = os.path.join("modelos", "best_gatos_classifier.pth")
            if not os.path.exists(model_path):
                print("\nAVISO: O modelo ainda não foi treinado ou salvo. Por favor, treine o modelo primeiro (Opção 1).")
                _ = input("Pressione Enter para continuar...")
            run_script('testar_gatos_desconhecidos.py') # 
        elif choice == '0':
            print("Saindo do programa. Até mais!")
            break
        else:
            print("Escolha inválida. Por favor, digite um número entre 0 e 3.")
            _ = input("Pressione Enter para continuar...") # Pausa para o usuário ler a mensagem

if __name__ == '__main__':
    main()