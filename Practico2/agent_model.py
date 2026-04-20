from agent import Agent


class AgentModel(Agent):
    """
    Agente basado en Modelo con Objetivo y Utilidad (Model-Based Goal/Utility Agent).

    Modelo interno:
        Estados discretos: MUY_FRIO, FRIO, COMODO, CALOR, MUY_CALOR
        El agente estima la tendencia de la temperatura externa para anticipar
        cómo evolucionará el ambiente en el siguiente paso.

    Objetivo (Goal):
        Mantener la temperatura en 0°C (estado COMODO).

    Utilidad:
        U(temp) = -|temp - 0|
        Cuanto más cerca de 0°C, mayor la utilidad.

    Diferencia con AgentReflex:
        El AgentReflex solo reacciona al estado actual (condición if-else).
        Este agente predice el estado SIGUIENTE usando el modelo interno y
        elige la acción que maximiza la utilidad de ese estado predicho.
    """

    GOAL = 0  # temperatura objetivo en grados

    # Definición de estados discretos y sus rangos (extiende el modelo Frio/Calor del enunciado)
    STATES = {
        'MUY_FRIO': (-float('inf'), -9),
        'FRIO':     (-9,            -2),
        'COMODO':   (-2,             2),
        'CALOR':    ( 2,             9),
        'MUY_CALOR':( 9,  float('inf')),
    }

    # Utilidad por estado (para logging y análisis)
    STATE_UTILITY = {
        'MUY_FRIO':  -15,
        'FRIO':      -5,
        'COMODO':     0,
        'CALOR':     -5,
        'MUY_CALOR': -15,
    }

    def __init__(self, env):
        self.env = env
        self.prev_obs = None        # observación anterior (estado interno)
        self.last_action = 0        # última acción tomada (para estimar tendencia)
        self.estimated_trend = 0    # tendencia externa estimada por el modelo

    def _discretize(self, obs):
        """Mapea temperatura continua al estado discreto correspondiente."""
        for state, (low, high) in self.STATES.items():
            if low <= obs < high:
                return state
        return 'MUY_CALOR'

    def _utility(self, temp):
        """
        Función de utilidad continua.
        Retorna valor más alto cuanto más cerca de 0°C.
        """
        return -abs(temp - self.GOAL)

    def _update_model(self, obs):
        """
        Actualiza el modelo interno estimando la tendencia de temperatura externa.

        Razonamiento:
            cambio_total = obs - prev_obs
            cambio_total = accion_propia + tendencia_externa
            => tendencia_externa = cambio_total - accion_propia
        """
        if self.prev_obs is not None:
            self.estimated_trend = obs - self.prev_obs - self.last_action

    def _predict_next_temp(self, obs, action):
        """
        Predice la temperatura del siguiente paso usando el modelo interno.

        temp_siguiente ≈ temp_actual + acción + tendencia_externa_estimada
        """
        return obs + action + self.estimated_trend

    def next_action(self, obs):
        """
        Elige la acción que maximiza la utilidad del estado predicho.

        Para cada acción posible:
          1. Predice la temperatura siguiente con el modelo
          2. Calcula la utilidad de esa temperatura predicha
          3. Elige la acción con mayor utilidad
        """
        self._update_model(obs)

        actions = range(
            self.env.action_space.start,
            self.env.action_space.start + self.env.action_space.n
        )

        best_action = 0
        best_utility = float('-inf')

        for action in actions:
            predicted_temp = self._predict_next_temp(obs, action)
            u = self._utility(predicted_temp)
            if u > best_utility:
                best_utility = u
                best_action = action

        # Actualiza estado interno para el próximo paso
        self.prev_obs = obs
        self.last_action = best_action
        return best_action
