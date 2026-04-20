# Práctico 3 — Explicación detallada de la solución

## Índice
1. [Entendiendo el problema](#1-entendiendo-el-problema)
2. [Cómo funciona el entorno (lo que ya estaba hecho)](#2-cómo-funciona-el-entorno-lo-que-ya-estaba-hecho)
3. [Agente UCS — paso a paso](#3-agente-ucs--paso-a-paso)
4. [Agente A* — paso a paso](#4-agente-a--paso-a-paso)
5. [La función main() — comparativa](#5-la-función-main--comparativa)
6. [Conceptos clave para que lo resuelvas solo](#6-conceptos-clave-para-que-lo-resuelvas-solo)

---

## 1. Entendiendo el problema

Tenemos un laberinto (grilla) con celdas:
- **S** = inicio
- **G** = destino (goal)
- **F** = hielo seguro (frozen, podés caminar)
- **H** = agujero (hole, si pisás perdés)

El agente tiene 4 acciones posibles:
- 0 = izquierda
- 1 = abajo
- 2 = derecha
- 3 = arriba

**Objetivo:** encontrar el camino más corto de S a G sin pisar H.

Esto es un **problema de búsqueda en grafos**, que es uno de los temas fundamentales de IA.

---

## 2. Cómo funciona el entorno (lo que ya estaba hecho)

### model.py — El grafo

El `Model` convierte la grilla 2D en un **grafo** representado como diccionario de diccionarios:

```python
graph[estado][accion] = estado_siguiente
```

Cada celda se identifica con un número entero (estado):
```
estado = fila * num_columnas + columna
```

Ejemplo en un mapa 4x4:
```
Celda (0,0) = estado 0    Celda (0,1) = estado 1    ...
Celda (1,0) = estado 4    Celda (1,1) = estado 5    ...
Celda (2,0) = estado 8    ...
```

**Reglas del grafo:**
- Solo las celdas S y F tienen aristas (acciones posibles)
- Las celdas H y G tienen `graph[estado] = {}` (diccionario vacío, no tienen salida)
- Si una acción te llevaría fuera del mapa, te quedás en el mismo estado

### priority_queue.py — La cola de prioridad

Almacena tuplas `(dato, costo, previo)` ordenadas por costo ascendente:
- `push(dato, costo, previo)` — inserta manteniendo el orden
- `pop()` — saca el elemento de menor costo
- `update(dato, nuevo_costo, previo)` — actualiza el costo de un elemento
- `dato in pq` — verifica si un dato está en la cola

### search_agent.py — Clase base

El flujo de ejecución es:
```
run() → _loop() → resetear entorno → repetir: _next_action() → env.step(accion)
```

La clase base lee acciones del teclado. Nosotros sobreescribimos `_next_action()` para que el agente decida solo.

---

## 3. Agente UCS — paso a paso

### ¿Qué es UCS?

**Uniform Cost Search** (Búsqueda de Costo Uniforme) es un algoritmo que siempre expande el nodo con **menor costo acumulado** desde el inicio. Garantiza encontrar el camino óptimo.

En este problema, como cada movimiento cuesta 1, UCS se comporta como BFS (Breadth-First Search). Pero la implementación con cola de prioridad es más general y funciona con costos variables.

### El algoritmo explicado

```
1. Meter el estado inicial en la cola de prioridad con costo 0
2. Mientras la cola no esté vacía:
   a. Sacar el estado de menor costo
   b. Si ya lo visitamos, saltear
   c. Marcarlo como visitado
   d. Si es el goal → reconstruir el camino y terminar
   e. Para cada acción posible desde este estado:
      - Obtener el estado siguiente
      - Si es un agujero (H) → saltear
      - Si es el mismo estado (rebote contra pared) → saltear
      - Si encontramos un camino más barato → actualizar y meter en la cola
```

### Estructuras de datos que usamos

1. **`PriorityQueue`** — para saber cuál nodo expandir primero (el de menor costo)
2. **`visited`** (set) — para no procesar un estado dos veces
3. **`came_from`** (dict) — para cada estado, guarda `(estado_previo, accion)`. Esto nos permite reconstruir el camino al final
4. **`cost_so_far`** (dict) — el mejor costo conocido para llegar a cada estado

### ¿Cómo evitamos los agujeros (H)?

Las celdas H tienen `graph[estado] = {}` (sin acciones salientes), igual que G. Para distinguirlas:
```python
if not self.model.graph[next_state] and next_state != self.end_state:
    continue  # es un agujero, no ir ahí
```

Si el diccionario de acciones está vacío Y no es el goal → es un agujero → lo salteamos.

### ¿Cómo reconstruimos el camino?

Cuando llegamos al goal, caminamos hacia atrás por `came_from`:

```
Goal ← came_from[goal] = (estado_X, accion_3)
estado_X ← came_from[estado_X] = (estado_Y, accion_1)
estado_Y ← came_from[estado_Y] = (estado_Z, accion_2)
estado_Z ← came_from[estado_Z] = None  (es el inicio)
```

Esto nos da las acciones en orden inverso: `[accion_3, accion_1, accion_2]`
Las revertimos: `[accion_2, accion_1, accion_3]` → este es el camino del inicio al goal.

### ¿Cómo funciona _next_action()?

La primera vez que se llama, `action_list` está vacía, así que ejecuta `ucs()` que calcula todo el camino de una vez. Luego, cada llamada simplemente saca la siguiente acción de la lista:

```python
def _next_action(self):
    if not self.action_list:
        self.ucs()          # calcular camino completo
    return self.action_list.pop(0)  # devolver siguiente acción
```

---

## 4. Agente A* — paso a paso

### ¿Qué es A*?

A* es una mejora de UCS que usa una **heurística** (estimación) para guiar la búsqueda. En vez de expandir solo por costo acumulado, usa:

```
f(n) = g(n) + h(n)
```

- **g(n)** = costo real desde el inicio hasta n (igual que UCS)
- **h(n)** = estimación del costo desde n hasta el goal (heurística)
- **f(n)** = estimación del costo total del camino pasando por n

La heurística "guía" la búsqueda hacia el goal, haciendo que explore menos nodos.

### Requisito de la heurística: admisibilidad

Para que A* garantice encontrar el camino óptimo, la heurística debe ser **admisible**: nunca debe sobreestimar el costo real. Es decir, `h(n) ≤ costo_real(n, goal)` siempre.

### Heurística 1: Distancia Manhattan

```
h(n) = |fila_n - fila_goal| + |col_n - col_goal|
```

Es la suma de las distancias horizontales y verticales. En una grilla donde solo te movés en 4 direcciones, es la menor cantidad de pasos posible (sin obstáculos).

**¿Es admisible?** Sí, porque nunca podés llegar al goal en menos pasos que la distancia Manhattan (ya que no podés moverte en diagonal).

**Ejemplo:** Si estás en (1,1) y el goal está en (3,3):
```
h = |1-3| + |1-3| = 2 + 2 = 4
```
Necesitás al menos 4 pasos (2 abajo + 2 derecha).

### Heurística 2: Distancia Euclidiana

```
h(n) = √((fila_n - fila_goal)² + (col_n - col_goal)²)
```

Es la distancia "en línea recta" (como si pudieras ir en diagonal).

**¿Es admisible?** Sí, porque la línea recta siempre es menor o igual que cualquier camino por la grilla.

**Ejemplo:** Mismo caso (1,1) a (3,3):
```
h = √(4 + 4) = √8 ≈ 2.83
```

### Manhattan vs Euclidiana: ¿cuál es mejor?

**Manhattan es mejor** porque es más "informativa" (da valores más altos, más cercanos al costo real). Cuanto más se acerque h(n) al costo real sin pasarse, menos nodos explora A*.

- Manhattan: h = 4 (más cerca del real)
- Euclidiana: h = 2.83 (más lejos del real)

Ambas son admisibles, pero Manhattan "poda" más el espacio de búsqueda.

### El algoritmo A* vs UCS

La única diferencia con UCS es la prioridad en la cola:

| Algoritmo | Prioridad en la cola |
|-----------|---------------------|
| UCS       | `g(n)` (costo acumulado) |
| A*        | `g(n) + h(n)` (costo acumulado + heurística) |

El resto del algoritmo (visited, came_from, reconstrucción) es idéntico.

### _calculate_heuristics()

Se ejecuta una sola vez en el constructor. Pre-calcula h(n) para TODOS los estados y los guarda en `self.heuristics[estado]`. Así durante la búsqueda solo hacemos `self.heuristics[next_state]` (O(1)).

Para calcular las coordenadas de un estado:
```python
fila = estado // num_columnas
columna = estado % num_columnas
```

---

## 5. La función main() — comparativa

### Estructura

```python
envs = [4x4, 8x8, 16x16, 32x32]  # 4 ambientes
agents = [UCS, A* Manhattan, A* Euclidean]  # 3 agentes
```

Para cada combinación (ambiente, agente):
1. Crear el modelo (grafo) del mapa
2. Encontrar estados S y G en el mapa
3. Crear el agente y cronometrar su ejecución
4. Registrar: reward (1.0 = llegó, 0.0 = no), pasos, tiempo

### Encontrar S y G

Recorremos el mapa celda por celda buscando 'S' y 'G':
```python
for fila in range(nrows):
    for col in range(ncols):
        celda = desc[fila, col].decode('utf-8')
        if celda == 'S':
            initial_state = fila * ncols + col
        elif celda == 'G':
            end_state = fila * ncols + col
```

### ¿Por qué try/except?

Los mapas son aleatorios. Puede pasar que los agujeros (H) bloqueen todos los caminos de S a G. En ese caso, el algoritmo termina sin encontrar solución y `action_list` queda vacía, causando un error. El try/except captura eso y muestra "SIN SOLUCION".

---

## 6. Conceptos clave para que lo resuelvas solo

### Para implementar UCS necesitás entender:
1. **Cola de prioridad**: siempre sacás el de menor costo
2. **Set de visitados**: nunca procesás un nodo dos veces
3. **Diccionario came_from**: para reconstruir el camino al final
4. **Filtrar H cells**: no expandir hacia agujeros

### Para implementar A* necesitás entender:
1. **Todo lo de UCS** (A* es UCS + heurística)
2. **f(n) = g(n) + h(n)**: la prioridad ahora incluye la estimación
3. **Admisibilidad**: la heurística nunca sobreestima
4. **Manhattan vs Euclidiana**: ambas admisibles, Manhattan más informativa

### Patrón general (aplica a ambos):

```
1. Inicializar cola con estado inicial
2. Mientras cola no vacía:
   a. Sacar nodo de menor prioridad
   b. Si ya visitado → skip
   c. Marcar visitado
   d. Si es goal → reconstruir camino
   e. Para cada vecino:
      - Filtrar inválidos (H, paredes)
      - Si mejora el costo → actualizar y encolar
3. Reconstruir camino caminando came_from de goal a inicio
4. Revertir la lista de acciones
```

### Diferencia clave entre UCS y A*:

```
UCS:  pq.push(next_state, g(n), ...)           # solo costo real
A*:   pq.push(next_state, g(n) + h(n), ...)    # costo real + estimación
```

Esa es literalmente la única línea que cambia en el algoritmo.
