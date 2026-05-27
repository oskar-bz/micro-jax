# micro-jax — Systemspezifikation

> Sprach- und implementierungsunabhängige Beschreibung des Systems.  
> Was das System ist, nicht wie oder wann es gebaut wird.

---

## 0. Grundprinzip

micro-jax ist ein System aus **funktionalen Transforms über reine Funktionen**.

Ein Transform nimmt eine Funktion und gibt eine neue Funktion zurück.
Die zurückgegebene Funktion hat dieselbe Aufruf-Signatur wie die Eingabe —
oder eine systematisch abgeleitete Variante davon.

```
Transform : (Array... → Array) → (Array... → Array)
```

Transforms sind first-class und komponierbar. Das einzige Mittel mit dem
das System Computation analysiert ist **Tracing**: die Eingabefunktion wird
mit abstrakten Werten ausgeführt, die Operationen aufzeichnen statt sie
auszuführen. Das Ergebnis des Tracings ist ein **Tape**.

Alle anderen Eigenschaften des Systems folgen aus diesen zwei Entscheidungen.

---

## 1. Basistypen

### 1.1 Skalar

Ein Skalar ist ein Element vom Typ `f32`.

### 1.2 NDArray

Ein NDArray ist ein rechteckiges, dicht gespeichertes Feld von Skalaren.

**Felder:**

| Feld | Typ | Beschreibung |
|---|---|---|
| `data` | `f32[]` | Flacher Speicherbereich der Werte |
| `shape` | `i32[ndim]` | Ausdehnung entlang jeder Dimension |
| `strides` | `i32[ndim]` | Schrittweite in Elementen pro Dimension |
| `ndim` | `i32` | Anzahl Dimensionen |
| `size` | `i32` | Produkt aller shape-Einträge |
| `id` | `i32` | Systemweit eindeutiger Bezeichner; unveränderlich nach Allokation |

**Sonderfälle:**

- `ndim = 0`: Skalar-Array. `shape = []`, `strides = []`, `size = 1`.
- `ndim = 1`: Vektor der Länge `shape[0]`.

**Index-Abbildung:**

Für einen Multi-Index `[i₀, i₁, ..., i_{n-1}]` ist der Flat-Index:

$$\text{flat} = \sum_{k=0}^{n-1} i_k \cdot \text{strides}[k]$$

Standard-Strides (row-major): `strides[k] = product(shape[k+1 .. n-1])`.

**Invarianten:**

1. `id` ist eindeutig über die gesamte Laufzeit des Systems.
2. `size = product(shape)`.
3. `len(data) >= size`.
4. Zwei NDArrays dürfen denselben `data`-Puffer teilen (Views), sofern ihre Zugriffsmuster disjunkt sind oder beide nur lesend zugreifen.

---

## 2. Primitive

### 2.1 Definition

Eine Primitive ist die atomare Recheneinheit des Systems. Jede Primitive
besteht aus drei Teilen:

```
Primitive = {
    name    : String
    fwd     : (f32[]...) → f32[]       -- Vorwärtsauswertung
    vjp     : (f32[], f32[]...) → f32[][]  -- VJP-Regel
}
```

`fwd` nimmt konkrete Wert-Arrays und gibt einen Wert-Array zurück.

`vjp` nimmt einen **Cotangent** (Gradient vom Output) sowie die gespeicherten
Primal-Werte der Inputs, und gibt für jeden Input einen Cotangent zurück.

### 2.2 VJP-Semantik

Sei $f$ eine Primitive mit Inputs $x_1, ..., x_n$ und Output $y$.
Sei $\bar{y}$ der eingehende Cotangent ($\partial L / \partial y$ für irgendeinen
skalaren Loss $L$).

Die VJP-Regel berechnet für jeden Input $x_i$:

$$\bar{x}_i = \bar{y} \cdot \frac{\partial y}{\partial x_i}$$

Für vektorwertige Operationen ist das die **Vektor-Jacobi-Produkt**-Form —
kein explizites Aufstellen der Jacobi-Matrix.

### 2.3 Primitive-Registry

Das System führt eine Registry aller bekannten Primitive.
Die Registry ist dynamisch erweiterbar: Nutzer können eigene Primitive
mit eigenem `fwd` und `vjp` registrieren.

Eingebaute Primitive:

| Name | fwd | vjp (für Input x, y) |
|---|---|---|
| `add` | $x + y$ | $(\bar{z},\ \bar{z})$ |
| `mul` | $x \cdot y$ | $(\bar{z} \cdot y,\ \bar{z} \cdot x)$ |
| `neg` | $-x$ | $(-\bar{z})$ |
| `sin` | $\sin(x)$ | $(\bar{z} \cdot \cos(x))$ |
| `exp` | $e^x$ | $(\bar{z} \cdot e^x)$ |
| `log` | $\ln(x)$ | $(\bar{z} / x)$ |
| `pow` | $x^n$ | $(\bar{z} \cdot n \cdot x^{n-1})$ |
| `matmul` | $A \cdot B$ | $(\bar{Z} \cdot B^T,\ A^T \cdot \bar{Z})$ |

Alle elementweisen Primitive operieren unabhängig entlang aller Dimensionen.
Shape des Outputs = Shape des Inputs (nach Broadcasting).

### 2.4 Broadcasting

Zwei NDArrays sind broadcast-kompatibel wenn ihre Shapes von rechts nach
links komponentenweise entweder gleich sind oder einer der Werte 1 ist.
Das Ergebnis-Shape ergibt sich als elementweises Maximum.

---

## 3. Tracer

### 3.1 Zweck

Ein Tracer ist ein NDArray das zusätzlich einem aktiven **Trace-Level** zugeordnet ist.
Er verhält sich nach außen wie ein normales NDArray — ist aber kein konkreter Wert,
sondern ein **Symbol** das eine Rechenoperation innerhalb eines Traces repräsentiert.

```
Tracer = NDArray ∪ { level : i32 }
```

### 3.2 Verhalten

Wenn eine Primitive auf Tracer-Inputs angewandt wird:

1. Der Vorwärts-Wert wird mit den konkreten Werten der Inputs berechnet.
2. Ein **TapeEntry** wird auf das aktive Tape des entsprechenden Levels geschrieben.
3. Der Rückgabewert ist ein neuer Tracer, der den Output repräsentiert.

Wenn eine Primitive auf gewöhnliche NDArray-Inputs angewandt wird (kein Tape aktiv):
Sie wird sofort ausgewertet und gibt ein gewöhnliches NDArray zurück.

### 3.3 `_emit`

Die interne Funktion `_emit` ist der einzige Ort im System der entscheidet
ob eine Primitive sofort ausgewertet oder aufgezeichnet wird.

```
_emit(prim, inputs[]):
    out_val  = prim.fwd(inputs[].val)
    out      = Tracer(val=out_val, level=max(inputs[].level))

    for each active tape T at level <= out.level:
        T.record(prim, inputs, out)

    return out
```

**Invariante:** `_emit` ist die einzige Funktion die auf ein Tape schreibt.

---

## 4. Tape

### 4.1 Struktur

Ein Tape ist eine geordnete Folge von TapeEntries.

```
Tape = TapeEntry[]

TapeEntry = {
    prim_id      : PrimID
    input_ids    : i32[]          -- IDs der Input-Tracer
    input_vals   : f32[][]        -- Kopien der Primal-Werte zur Zeit des Forward Pass
    output_id    : i32            -- ID des Output-Tracers
    output_shape : i32[]
}
```

### 4.2 Eigenschaften

- Einträge sind chronologisch geordnet: Eintrag $i$ wurde vor Eintrag $j > i$ aufgezeichnet.
- `input_vals` sind **Kopien**, keine Referenzen. Sie sind unabhängig von
  späteren Veränderungen an den Originalarrays.
- Ein TapeEntry enthält alle Information die für den Reverse Pass nötig ist —
  ohne erneutes Ausführen des Forward Pass.

### 4.3 Trace-Stack

Das System führt einen **Stack aktiver Tapes**, einem pro aktivem Transform.
Jedes Tape hat ein **Level** — seine Position im Stack zum Zeitpunkt des Push.

```
TraceStack = Tape[]   -- Stack, Level 0 ist das äußerste Tape
```

`_emit` schreibt auf alle Tapes deren Level ≤ Level des Outputs.
Das ermöglicht verschachtelte Transforms.

**Invariante:** Der Stack ist leer wenn kein Transform aktiv ist.
In diesem Zustand führt `_emit` alle Operationen eager aus.

---

## 5. Transforms

### 5.1 `grad`

**Signatur:**
```
grad(f, argnum) : (Array... → Array) → (Array... → Array)
```

**Semantik:**

`grad(f, argnum)` gibt eine Funktion zurück die, wenn mit denselben Argumenten
wie `f` aufgerufen, den Gradienten von `f`'s skalarem Output bezüglich des
`argnum`-ten Arguments zurückgibt.

**Voraussetzung:** `f` hat einen skalaren Output (ndim=0, size=1).

**Algorithmus:**

```
grad(f, argnum)(args[]):

    -- FORWARD: f mit Tracern ausführen, Tape aufbauen
    tape  = new Tape()
    push(tape)
    tracers = [Tracer(a) for a in args]
    out   = f(tracers...)
    pop()

    -- REVERSE: Cotangents rückwärts propagieren
    cotangents = Map<ID, Array>
    cotangents[out.id] = ones_like(out)    -- Seed: dL/dL = 1

    for entry in reversed(tape):
        g = cotangents.get(entry.output_id, zeros_like(output))
        input_grads = entry.prim.vjp(g, entry.input_vals...)

        for (id, grad) in zip(entry.input_ids, input_grads):
            cotangents[id] += cotangents.get(id, zeros) + grad

    return cotangents[tracers[argnum].id]
```

**Eigenschaften:**

- `grad(f, argnum)` ist selbst eine reine Funktion — kein globaler State nach dem Aufruf.
- `grad(grad(f))` ist gültig: erfordert verschachtelte Tapes und Trace-Levels (§4.3).
- Der zurückgegebene Gradient hat dieselbe Shape wie `args[argnum]`.
- Cotangents werden über alle Fanout-Pfade akkumuliert (`+=`).

### 5.2 `jit`

**Signatur:**
```
jit(f) : (Array... → Array) → (Array... → Array)
```

**Semantik:**

`jit(f)` gibt eine Funktion zurück die sich bei gleichem Shape-Muster der Inputs
wie `f` verhält, aber nach dem ersten Aufruf eine zwischengespeicherte kompilierte
Version verwendet.

**Cache-Schlüssel:**

```
ShapeSignature = [(ndim, shape[], dtype) for each input]
```

Zwei Aufrufe mit identischer `ShapeSignature` verwenden denselben kompilierten Pfad.

**Abstract Tracing:**

Beim ersten Aufruf mit einer neuen Signatur wird `f` **abstract traced**:
Inputs werden durch AbstractValues ersetzt — sie tragen nur Shape und Dtype,
keine konkreten Werte. Das Ergebnis ist ein **AbstractTape**: eine Folge von
Operationen mit bekannten Input/Output-Shapes, aber ohne Werte.

```
AbstractValue = { shape : i32[], dtype : Dtype }
AbstractTape  = AbstractTapeEntry[]

AbstractTapeEntry = {
    prim_id      : PrimID
    input_shapes : Shape[]
    output_shape : Shape
}
```

**Optimierungspässe auf AbstractTape:**

Das AbstractTape wird vor der Ausführung durch drei Pässe verarbeitet:

1. **Dead Code Elimination (DCE):**
   Entferne alle Einträge deren Output nicht (direkt oder transitiv) als Input
   des finalen Outputs dient.

2. **Constant Folding:**
   Einträge deren alle Inputs konstant sind (nicht von Trace-Inputs abhängen)
   werden sofort ausgewertet; ihr Output-Wert wird in den Tape eingebettet.

3. **Common Subexpression Elimination (CSE):**
   Zwei Einträge mit identischer `prim_id` und identischen `input_ids`
   werden zu einem zusammengeführt.

**Invariante:** `jit(f)(x) = f(x)` für alle `x` mit derselben Shape-Signatur.

### 5.3 `vmap`

**Signatur:**
```
vmap(f, in_axes) : (Array... → Array) → (Array... → Array)
```

`in_axes` spezifiziert für jeden Input entlang welcher Achse gebatcht wird.
Default: Achse 0 für alle Inputs.

**Semantik:**

`vmap(f)(X)` entspricht `[f(X[i]) for i in range(N)]` gestackt entlang Achse 0,
aber ohne Python-Loop und ohne N-fachen Overhead.

**Mechanismus — BatchTracer:**

Jeder Input wird mit einem `batch_dim`-Tag versehen. Dieser Tag propagiert
durch alle Primitive via **Batch-Regeln**.

```
BatchTracer = Tracer ∪ { batch_dim : i32 | None }
```

Jede Primitive hat eine Batch-Regel:

```
BatchRule = (inputs[], batch_dims[]) → (output, out_batch_dim)
```

Die Batch-Regel beschreibt wie die Primitive mit gebatchten Inputs umgeht,
ohne einen expliziten Loop auszuführen. Typische Fälle:

- Beide Inputs gebatcht an Dim 0: elementweise Op, Output gebatcht an Dim 0.
- Ein Input gebatcht, einer nicht: broadcast den ungebatchten Input.
- Kein Input gebatcht: wird von `vmap` nicht aufgerufen.

**Invariante:** `vmap(f)(X)[i] = f(X[i])` für alle `i` in `range(X.shape[0])`.

---

## 6. Zusammensetzungsregeln

Transforms können beliebig verschachtelt werden. Die folgenden Kombinationen
sind explizit spezifiziert:

### `jit(grad(f))`

Der `grad`-Transform wird abstract traced. Das Tape des Reverse Pass wird
Teil des AbstractTape. Das kompilierte Programm enthält den vollständigen
Gradienten-Code ohne Python-Overhead.

### `vmap(grad(f))`

Für jeden Batch-Eintrag wird `grad(f)` ausgeführt. Der `grad`-Transform
läuft innerhalb des `vmap`-BatchTracers — d.h. alle Cotangent-Berechnungen
sind selbst gebatcht.

**Ergebnis:** Per-sample Gradienten in einem einzigen Aufruf.

### `grad(grad(f))`

Der äußere `grad` traced den inneren `grad`. Die Cotangent-Arithmetik
des inneren Reverse Pass ist sichtbar für das äußere Tape, weil sie
aus registrierten Primitiven besteht die `_emit` aufrufen.

**Voraussetzung:** Trace-Level-System (§4.3). Ohne Level-Tracking sieht
das äußere Tape die inneren Operationen nicht.

### Allgemeine Regel

Ein Transform $T_1$ kann einen Transform $T_2$ wrappen wenn:
- $T_2(f)$ eine Funktion zurückgibt die aus registrierten Primitiven besteht, und
- $T_1$ die Semantik von $T_2$ nicht verletzt (d.h. $T_1$ liest nur Tapes
  seines eigenen Levels).

---

## 7. Verträge und Invarianten

### 7.1 Reinheits-Vertrag

Jede Funktion die einem Transform übergeben wird muss folgende Eigenschaften haben:

**Keine wertabhängigen Seiteneffekte:** Sie darf keine externen Variablen
verändern, keine I/O durchführen, und keine Zustände lesen die sich zwischen
zwei Aufrufen mit gleichen Inputs ändern könnten.

**Keine Array-Mutation:** Sie darf Input-Arrays nicht in-place verändern.

**Kein wertabhängiger Kontrollfluss auf Tracern:** `if tracer > 0` ist
undefiniertes Verhalten. Der Pfad wird einmalig beim Tracing eingefroren
und nicht bei jedem Aufruf neu evaluiert.

*Anmerkung:* Das System kann diese Verträge nicht vollständig erzwingen.
Verletzungen führen zu stillem Fehlverhalten, nicht zu Laufzeitfehlern.
Debug-Modus-Assertions sollten die häufigsten Verletzungen erkennen.

### 7.2 Korrektheitsinvariante

Für jeden Transform $T$ und jede Funktion $f$ gilt:

$$T(f)(x) \text{ ist semantisch korrekt wenn } f \text{ den Reinheits-Vertrag erfüllt}$$

Insbesondere:

- `grad(f)(x)` stimmt mit dem analytischen Gradienten überein (bis auf numerische Präzision).
- `jit(f)(x) = f(x)` für alle `x`.
- `vmap(f)(X)[i] = f(X[i])` für alle `i`.

### 7.3 ID-Eindeutigkeit

Jeder NDArray erhält eine systemweit eindeutige `id` zum Zeitpunkt der Allokation.
Diese ID ist unveränderlich. Sie wird im Reverse Pass als Schlüssel für
Cotangent-Lookup verwendet.

**Invariante:** Zwei NDArrays mit gleichem `id` sind dasselbe Objekt.

### 7.4 Tape-Isolation

Ein Tape auf Level $l$ enthält ausschließlich Operationen die während der
Ausführung des Transforms auf Level $l$ emittiert wurden.

Operationen eines inneren Transforms (Level $< l$) sind auf dem Tape
von Level $l$ sichtbar — das ist die Voraussetzung für `grad(grad(f))`.

Operationen eines äußeren Transforms (Level $> l$) sind auf dem Tape
von Level $l$ **nicht** sichtbar.

---

## 8. Speicher-Modell

### 8.1 Ownership

Jeder NDArray hat genau einen Besitzer zu jedem Zeitpunkt.
Beim Übergeben an eine Funktion wird Ownership übertragen.
Nach dem Return einer Transform-Funktion gehört der Output dem Aufrufer.

### 8.2 Tape-Scoped Allokation

Alle während eines Trace allokierten Intermediate-Arrays gehören dem Tape.
Sie sind nach dem Ende des Traces ungültig, sofern nicht explizit persistiert.

**Persistierung:** Ein Output-Array das außerhalb des Trace-Scopes verwendet
werden soll muss explizit kopiert werden (Operation: `persist`).

### 8.3 Saved Input Values

`TapeEntry.input_vals` sind tiefe Kopien zum Zeitpunkt des Forward Pass.
Sie gehören dem TapeEntry und leben solange wie das Tape.

---

## 9. Nicht im System-Scope

Die folgenden Eigenschaften sind bewusst nicht Teil dieser Spezifikation:

| Feature | Begründung |
|---|---|
| GPU / Hardware-Backends | Erfordert Kernel-Infrastruktur orthogonal zu Tracing |
| `pmap` (Parallelismus) | Erfordert Multiprocess-Koordination |
| Pytrees als Eingabetypen | Orthogonales Feature, erweiterbar ohne Kern-Änderungen |
| Automatische Speicherverwaltung | Ownership-Modell ist explizit spezifiziert |
| Wertabhängiger Kontrollfluss (cond, while_loop) | Erfordert strukturelle Traces (Jaxprs), nicht flache Tapes |
| Dtype-Polymorphismus | Nur f32 |
| Zweite Ableitungen via JVP | Nur VJP (Reverse Mode) als primärer AD-Mechanismus |
