class ModbusClientHandler:
    def __init__(self, client):
        """
        Inizializza il gestore con un'istanza del client Modbus.
        """
        self.client = client
        self.commands = {
            "read_holding_registers": self.client.read_holding_registers,
            "write_register": self.client.write_register,
            "read_coils": self.client.read_coils,
            "write_coil": self.client.write_coil,
            "read_input_registers": self.client.read_input_registers
        }
    
    def execute_command(self, command_name, **kwargs):
        """
        Esegue un comando dinamicamente in base al nome e agli argomenti forniti.
        
        :param command_name: Nome del comando da eseguire (stringa).
        :param kwargs: Argomenti dinamici da passare al comando.
        :return: Risultato del comando o errore.
        """
        try:
            # Controlla se il comando esiste
            if command_name in self.commands:
                # Esegui il comando con gli argomenti forniti
                return self.commands[command_name](**kwargs)
            else:
                raise ValueError(f"Comando '{command_name}' non supportato.")
        except Exception as e:
            return f"Errore nell'esecuzione del comando '{command_name}': {e}"
        


