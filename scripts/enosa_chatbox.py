import sys
from pathlib import Path

# Añadir el root del proyecto al path para que encuentre tau2
sys.path.append(str(Path(__file__).parent.parent))

from tau2.domains.enosa_masias.data_model import get_db
from tau2.domains.enosa_masias.tools import EnosaToolKit

def main():
    print("=== ENOSA CHATBOX TESTER ===")
    db = get_db()
    tools = EnosaToolKit(db)
    
    while True:
        print("\nOpciones: 1. Info, 2. Deuda, 3. Buscar Suministros, 4. Crear Ticket, 5. Ver Ticket, q. Salir")
        choice = input("Selecciona una opción: ").strip().lower()
        
        try:
            if choice == '1':
                print(tools.get_enosa_info())
            elif choice == '2':
                sn = input("Número de suministro (ej. S-101): ")
                print(tools.get_supply_details(sn))
            elif choice == '3':
                dni = input("DNI del cliente: ")
                print(tools.search_supplies_by_dni(dni))
            elif choice == '4':
                dni = input("DNI: ")
                tipo = input("Tipo (power_outage/public_hazard/billing): ")
                desc = input("Descripción: ")
                sn = input("Suministro (opcional): ")
                print(tools.create_ticket(dni, tipo, desc, sn if sn else None))
            elif choice == '5':
                tid = input("ID del Ticket (ej. T-001): ")
                print(tools.get_ticket(tid))
            elif choice == 'q':
                break
            else:
                print("Opción inválida, mano.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()