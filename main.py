"""
====================================================================
MOTOR DE GERAÇÃO DE HORÁRIOS ESCOLARES
API FastAPI + Google OR-Tools CP-SAT Solver
====================================================================
Resolve o School Timetabling Problem usando Constraint Programming.
Recebe dados via POST JSON, retorna a grade otimizada.
====================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional, Union
from ortools.sat.python import cp_model
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Motor de Horários Escolares",
    description="API para geração otimizada de grades horárias usando CP-SAT",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# MODELOS DE DADOS
# ──────────────────────────────────────────────

class Professor(BaseModel):
    id: Union[str, int, float]
    nome: str

    @field_validator('id', mode='before')
    @classmethod
    def coerce_id(cls, v):
        return str(v)

class Turma(BaseModel):
    id: Union[str, int, float]
    nome: str
    serie: Union[str, int, float] = ""
    turno: Optional[str] = "Integral"

    @field_validator('id', 'serie', mode='before')
    @classmethod
    def coerce_str(cls, v):
        return str(v) if v is not None else ""

class Disciplina(BaseModel):
    id: Union[str, int, float]
    nome: str
    abreviacao: Optional[str] = ""

    @field_validator('id', mode='before')
    @classmethod
    def coerce_id(cls, v):
        return str(v)

class CargaHoraria(BaseModel):
    id: Union[str, int, float]
    turma_id: Union[str, int, float]
    disciplina_id: Union[str, int, float]
    professor_id: Union[str, int, float]
    aulas_semana: Union[int, str, float]
    geminada: Optional[str] = "NAO"

    @field_validator('id', 'turma_id', 'disciplina_id', 'professor_id', mode='before')
    @classmethod
    def coerce_str(cls, v):
        return str(v) if v is not None else ""

    @field_validator('aulas_semana', mode='before')
    @classmethod
    def coerce_aulas(cls, v):
        return int(float(v)) if v is not None else 0

class Indisponibilidade(BaseModel):
    professor_id: Union[str, int, float]
    dia: str
    slot: Union[int, str, float]

    @field_validator('professor_id', mode='before')
    @classmethod
    def coerce_str(cls, v):
        return str(v) if v is not None else ""

    @field_validator('slot', mode='before')
    @classmethod
    def coerce_slot(cls, v):
        return int(float(v)) if v is not None else 0

class InputData(BaseModel):
    professores: list[Professor]
    turmas: list[Turma]
    disciplinas: list[Disciplina]
    carga_horaria: list[CargaHoraria]
    indisponibilidades: list[Indisponibilidade]
    config: Optional[dict] = None

# ──────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────

DIAS = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
SLOTS_MANHA = [1, 2, 3, 4, 5]
SLOTS_TARDE = [6, 7, 8, 9]
TODOS_SLOTS = SLOTS_MANHA + SLOTS_TARDE
NUM_DIAS = len(DIAS)
NUM_SLOTS = len(TODOS_SLOTS)

# ──────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "online", "engine": "OR-Tools CP-SAT", "version": "1.0.0"}

# ──────────────────────────────────────────────
# ENDPOINT PRINCIPAL
# ──────────────────────────────────────────────

@app.post("/gerar-grade")
def gerar_grade(data: InputData):
    start_time = time.time()
    logger.info(f"Recebido: {len(data.turmas)} turmas, {len(data.professores)} professores, {len(data.carga_horaria)} cargas")

    try:
        resultado = resolver_horarios(data)
        elapsed = round(time.time() - start_time, 2)
        resultado["tempo_segundos"] = elapsed
        logger.info(f"Resolvido em {elapsed}s — status: {resultado['status']}")
        return resultado
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Erro no solver: {tb}")
        return {
            "status": "erro",
            "message": f"Erro interno no solver: {str(e)}",
            "traceback": tb,
            "alocacoes": [],
            "total_aulas": 0,
            "alocadas": 0,
            "nao_alocadas": 0,
            "tempo_segundos": round(time.time() - start_time, 2)
        }


def resolver_horarios(data: InputData) -> dict:
    """
    Modela e resolve o problema de timetabling usando CP-SAT.
    Usa índices numéricos para nomear variáveis (evita caracteres especiais dos IDs).
    """

    model = cp_model.CpModel()

    # ── Mapear IDs para índices numéricos (safe para nomes de variáveis) ──
    prof_list = [str(p.id) for p in data.professores]
    turma_list = [str(t.id) for t in data.turmas]

    # Indisponibilidades: set de (prof_id_str, dia_idx, slot_idx)
    indisp_set = set()
    dia_to_idx = {d: i for i, d in enumerate(DIAS)}
    slot_to_idx = {s: i for i, s in enumerate(TODOS_SLOTS)}

    for ind in data.indisponibilidades:
        d_idx = dia_to_idx.get(ind.dia)
        s_idx = slot_to_idx.get(int(ind.slot))
        if d_idx is not None and s_idx is not None:
            indisp_set.add((str(ind.professor_id), d_idx, s_idx))

    # ── Expandir cargas em "aulas" individuais ──
    aula_meta = []

    for c in data.carga_horaria:
        n = int(c.aulas_semana)
        is_gem = (c.geminada == "SIM")
        tid = str(c.turma_id)
        did = str(c.disciplina_id)
        pid = str(c.professor_id)

        if is_gem and n >= 2:
            for _ in range(n // 2):
                aula_meta.append({"turma_id": tid, "disc_id": did, "prof_id": pid, "tipo": "geminada"})
            if n % 2 == 1:
                aula_meta.append({"turma_id": tid, "disc_id": did, "prof_id": pid, "tipo": "simples"})
        else:
            for _ in range(n):
                aula_meta.append({"turma_id": tid, "disc_id": did, "prof_id": pid, "tipo": "simples"})

    num_aulas = len(aula_meta)
    logger.info(f"Total de aulas a alocar: {num_aulas}")

    if num_aulas == 0:
        return {
            "status": "ok", "message": "Nenhuma aula para alocar.",
            "alocacoes": [], "total_aulas": 0, "alocadas": 0, "nao_alocadas": 0
        }

    # ── Variáveis de decisão: x[a][d][s] ──
    # Para simples: x[a,d,s]=1 se aula a está no dia d slot s
    # Para geminada: x[a,d,s]=1 se aula a COMEÇA no dia d slot s (ocupa s e s+1)
    x = {}
    for a in range(num_aulas):
        for d in range(NUM_DIAS):
            for s in range(NUM_SLOTS):
                x[(a, d, s)] = model.new_bool_var(f"a{a}_d{d}_s{s}")

    # ── Pré-computar slots válidos para geminadas ──
    manha_len = len(SLOTS_MANHA)
    gem_valid_starts = set()
    for d in range(NUM_DIAS):
        for s in range(manha_len - 1):  # manhã
            gem_valid_starts.add((d, s))
        for s in range(manha_len, NUM_SLOTS - 1):  # tarde
            gem_valid_starts.add((d, s))

    # ══ HARD CONSTRAINT 1: Cada aula alocada exatamente 1 vez ══
    for a in range(num_aulas):
        meta = aula_meta[a]
        if meta["tipo"] == "geminada":
            valid = [x[(a, d, s)] for (d, s) in gem_valid_starts]
            invalid = [x[(a, d, s)] for d in range(NUM_DIAS) for s in range(NUM_SLOTS) if (d, s) not in gem_valid_starts]
            for v in invalid:
                model.add(v == 0)
            model.add_exactly_one(valid)
        else:
            all_vars = [x[(a, d, s)] for d in range(NUM_DIAS) for s in range(NUM_SLOTS)]
            model.add_exactly_one(all_vars)

    # ── Helper: quais variáveis "ocupam" slot (d,s) para aula a ──
    def ocupa_slot(a, d, s):
        """Retorna lista de variáveis x que fazem aula a ocupar dia d slot s."""
        meta = aula_meta[a]
        vars_occ = [x[(a, d, s)]]
        if meta["tipo"] == "geminada" and s > 0:
            # Geminada que COMEÇA em s-1 também ocupa s, se no mesmo turno
            s_prev = s - 1
            same_turno = (s_prev < manha_len and s < manha_len) or (s_prev >= manha_len and s >= manha_len)
            if same_turno:
                vars_occ.append(x[(a, d, s_prev)])
        return vars_occ

    # ══ HARD CONSTRAINT 2: Conflito de professor (max 1 por slot) ══
    prof_aulas = {}
    for a in range(num_aulas):
        pid = aula_meta[a]["prof_id"]
        prof_aulas.setdefault(pid, []).append(a)

    for pid, als in prof_aulas.items():
        if len(als) <= 1:
            continue
        for d in range(NUM_DIAS):
            for s in range(NUM_SLOTS):
                occ = []
                for a in als:
                    occ.extend(ocupa_slot(a, d, s))
                if len(occ) > 1:
                    model.add(sum(occ) <= 1)

    # ══ HARD CONSTRAINT 3: Conflito de turma (max 1 por slot) ══
    turma_aulas = {}
    for a in range(num_aulas):
        tid = aula_meta[a]["turma_id"]
        turma_aulas.setdefault(tid, []).append(a)

    for tid, als in turma_aulas.items():
        if len(als) <= 1:
            continue
        for d in range(NUM_DIAS):
            for s in range(NUM_SLOTS):
                occ = []
                for a in als:
                    occ.extend(ocupa_slot(a, d, s))
                if len(occ) > 1:
                    model.add(sum(occ) <= 1)

    # ══ HARD CONSTRAINT 4: Disponibilidade do professor ══
    for a in range(num_aulas):
        meta = aula_meta[a]
        pid = meta["prof_id"]
        for d in range(NUM_DIAS):
            for s in range(NUM_SLOTS):
                if (pid, d, s) in indisp_set:
                    model.add(x[(a, d, s)] == 0)
                # Geminada: se s+1 indisponível, não pode começar em s
                if meta["tipo"] == "geminada" and s + 1 < NUM_SLOTS:
                    if (pid, d, s + 1) in indisp_set:
                        model.add(x[(a, d, s)] == 0)

    # ══ SOFT CONSTRAINTS ══
    penalties = []

    # SOFT 1: Max 2 aulas da mesma disciplina/turma por dia
    turma_disc_aulas = {}
    for a in range(num_aulas):
        key = (aula_meta[a]["turma_id"], aula_meta[a]["disc_id"])
        turma_disc_aulas.setdefault(key, []).append(a)

    sc_idx = 0
    for (tid, did), als in turma_disc_aulas.items():
        for d in range(NUM_DIAS):
            day_vars = [x[(a, d, s)] for a in als for s in range(NUM_SLOTS)]
            if len(day_vars) > 2:
                excess = model.new_int_var(0, len(day_vars), f"sc{sc_idx}")
                sc_idx += 1
                model.add(excess >= sum(day_vars) - 2)
                penalties.append(excess * 50)

    # SOFT 2: Distribuir aulas ao longo da semana
    for (tid, did), als in turma_disc_aulas.items():
        if len(als) <= 1:
            continue
        for d in range(NUM_DIAS):
            day_vars = [x[(a, d, s)] for a in als for s in range(NUM_SLOTS)]
            if day_vars:
                over = model.new_int_var(0, 10, f"sc{sc_idx}")
                sc_idx += 1
                model.add(over >= sum(day_vars) - 1)
                penalties.append(over * 10)

    # SOFT 3: Evitar professor em todos os 5 dias (simplificado)
    for pid, als in prof_aulas.items():
        day_bools = []
        for d in range(NUM_DIAS):
            day_vars = [x[(a, d, s)] for a in als for s in range(NUM_SLOTS)]
            if day_vars:
                b = model.new_bool_var(f"sc{sc_idx}")
                sc_idx += 1
                # b=1 se professor tem pelo menos 1 aula nesse dia
                model.add(sum(day_vars) >= 1).only_enforce_if(b)
                model.add(sum(day_vars) == 0).only_enforce_if(b.Not())
                day_bools.append(b)
        if len(day_bools) == 5:
            # Penalizar se todos os 5 dias usados
            all5 = model.new_bool_var(f"sc{sc_idx}")
            sc_idx += 1
            model.add(sum(day_bools) >= 5).only_enforce_if(all5)
            model.add(sum(day_bools) <= 4).only_enforce_if(all5.Not())
            penalties.append(all5 * 30)

    # ── FUNÇÃO OBJETIVO ──
    if penalties:
        model.minimize(sum(penalties))

    # ── RESOLVER ──
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120
    solver.parameters.num_workers = 2  # free tier friendly
    solver.parameters.log_search_progress = False

    logger.info(f"Iniciando solver: {num_aulas} aulas, {len(data.turmas)} turmas, {len(prof_aulas)} profs")
    status = solver.Solve(model)

    status_map = {
        cp_model.OPTIMAL: "otimo",
        cp_model.FEASIBLE: "viavel",
        cp_model.INFEASIBLE: "inviavel",
        cp_model.MODEL_INVALID: "modelo_invalido",
        cp_model.UNKNOWN: "desconhecido"
    }
    result_status = status_map.get(status, "desconhecido")

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return {
            "status": result_status,
            "message": "Não foi possível encontrar solução viável. Verifique se as restrições não são conflitantes.",
            "alocacoes": [], "total_aulas": num_aulas, "alocadas": 0, "nao_alocadas": num_aulas,
            "detalhes_solver": {"wall_time": round(solver.WallTime(), 2)}
        }

    # ── EXTRAIR SOLUÇÃO ──
    alocacoes = []
    for a in range(num_aulas):
        meta = aula_meta[a]
        found = False
        for d in range(NUM_DIAS):
            for s in range(NUM_SLOTS):
                if solver.Value(x[(a, d, s)]) == 1:
                    alocacoes.append({
                        "turma_id": meta["turma_id"],
                        "disciplina_id": meta["disc_id"],
                        "professor_id": meta["prof_id"],
                        "dia": DIAS[d],
                        "slot": TODOS_SLOTS[s]
                    })
                    if meta["tipo"] == "geminada" and s + 1 < NUM_SLOTS:
                        alocacoes.append({
                            "turma_id": meta["turma_id"],
                            "disciplina_id": meta["disc_id"],
                            "professor_id": meta["prof_id"],
                            "dia": DIAS[d],
                            "slot": TODOS_SLOTS[s + 1]
                        })
                    found = True
                    break
            if found:
                break

    total_slots = len(alocacoes)
    pen_total = round(solver.ObjectiveValue(), 1) if penalties else 0

    return {
        "status": result_status,
        "message": f"Grade gerada com sucesso! {total_slots} aulas alocadas.",
        "alocacoes": alocacoes,
        "total_aulas": num_aulas,
        "alocadas": total_slots,
        "nao_alocadas": 0,
        "penalidade_total": pen_total,
        "detalhes_solver": {
            "wall_time": round(solver.WallTime(), 2),
            "objective": pen_total
        }
    }
