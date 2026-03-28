"""
====================================================================
MOTOR DE GERAÇÃO DE HORÁRIOS ESCOLARES v2
API FastAPI + Google OR-Tools CP-SAT Solver
====================================================================
Estratégia: Decomposição por turma com rastreamento global de
conflitos de professor. Resolve cada turma como um sub-problema
CP-SAT pequeno, respeitando alocações anteriores.
Memória: ~50MB para 50 turmas (cabe no free tier 512MB).
====================================================================
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional, Union
from ortools.sat.python import cp_model
import time
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Motor de Horarios Escolares", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ──────────────────────────────────────────────
# MODELOS DE DADOS (tolerantes a tipos do Sheets)
# ──────────────────────────────────────────────

class Professor(BaseModel):
    id: Union[str, int, float]
    nome: str
    @field_validator('id', mode='before')
    @classmethod
    def _sid(cls, v): return str(v)

class Turma(BaseModel):
    id: Union[str, int, float]
    nome: str
    serie: Union[str, int, float] = ""
    turno: Optional[str] = "Integral"
    @field_validator('id', 'serie', mode='before')
    @classmethod
    def _sid(cls, v): return str(v) if v is not None else ""

class Disciplina(BaseModel):
    id: Union[str, int, float]
    nome: str
    abreviacao: Optional[str] = ""
    @field_validator('id', mode='before')
    @classmethod
    def _sid(cls, v): return str(v)

class CargaHoraria(BaseModel):
    id: Union[str, int, float]
    turma_id: Union[str, int, float]
    disciplina_id: Union[str, int, float]
    professor_id: Union[str, int, float]
    aulas_semana: Union[int, str, float]
    geminada: Optional[str] = "NAO"
    @field_validator('id', 'turma_id', 'disciplina_id', 'professor_id', mode='before')
    @classmethod
    def _sid(cls, v): return str(v) if v is not None else ""
    @field_validator('aulas_semana', mode='before')
    @classmethod
    def _int(cls, v): return int(float(v)) if v is not None else 0

class Indisponibilidade(BaseModel):
    professor_id: Union[str, int, float]
    dia: str
    slot: Union[int, str, float]
    @field_validator('professor_id', mode='before')
    @classmethod
    def _sid(cls, v): return str(v) if v is not None else ""
    @field_validator('slot', mode='before')
    @classmethod
    def _int(cls, v): return int(float(v)) if v is not None else 0

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

DIAS = ["Segunda", "Terca", "Quarta", "Quinta", "Sexta"]
DIAS_ACCEPT = {
    "Segunda": 0, "segunda": 0,
    "Terça": 1, "terça": 1, "Terca": 1, "terca": 1, "Terca": 1,
    "Quarta": 2, "quarta": 2,
    "Quinta": 3, "quinta": 3,
    "Sexta": 4, "sexta": 4,
}
SLOTS_MANHA = [1, 2, 3, 4, 5]
SLOTS_TARDE = [6, 7, 8, 9]
TODOS_SLOTS = SLOTS_MANHA + SLOTS_TARDE
NUM_DIAS = len(DIAS)
NUM_SLOTS = len(TODOS_SLOTS)
MANHA_LEN = len(SLOTS_MANHA)
SLOT_TO_IDX = {s: i for i, s in enumerate(TODOS_SLOTS)}
DIA_NAMES = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]  # nomes bonitos p/ output

@app.get("/")
def health():
    return {"status": "online", "engine": "OR-Tools CP-SAT v2 (decomposed)", "version": "2.0.0"}

# ──────────────────────────────────────────────
# ENDPOINT PRINCIPAL
# ──────────────────────────────────────────────

@app.post("/gerar-grade")
def gerar_grade(data: InputData):
    start_time = time.time()
    logger.info(f"Recebido: {len(data.turmas)} turmas, {len(data.professores)} profs, {len(data.carga_horaria)} cargas")
    try:
        resultado = resolver_horarios(data)
        resultado["tempo_segundos"] = round(time.time() - start_time, 2)
        logger.info(f"Pronto em {resultado['tempo_segundos']}s - {resultado['status']}")
        return resultado
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Erro: {tb}")
        return {
            "status": "erro",
            "message": f"Erro interno: {str(e)}",
            "traceback": tb,
            "alocacoes": [], "total_aulas": 0, "alocadas": 0, "nao_alocadas": 0,
            "tempo_segundos": round(time.time() - start_time, 2)
        }

# ──────────────────────────────────────────────
# SOLVER DECOMPOSTO POR TURMA
# ──────────────────────────────────────────────

def resolver_horarios(data: InputData) -> dict:
    """
    Resolve turma por turma. Cada sub-problema tem ~20-40 vars.
    Rastreia conflitos de professor globalmente.
    """

    # Indisponibilidades: set(prof_id, dia_i, slot_i)
    indisp = set()
    for ind in data.indisponibilidades:
        di = DIAS_ACCEPT.get(ind.dia)
        si = SLOT_TO_IDX.get(int(ind.slot))
        if di is not None and si is not None:
            indisp.add((str(ind.professor_id), di, si))

    logger.info(f"Indisponibilidades carregadas: {len(indisp)}")

    # Agrupar cargas por turma
    cargas_por_turma = {}
    for c in data.carga_horaria:
        tid = str(c.turma_id)
        cargas_por_turma.setdefault(tid, []).append(c)

    # Ocupação global do professor: set de (prof_id, dia_i, slot_i)
    prof_ocupado = set()

    # Ordenar: turmas com mais aulas primeiro (mais restritas)
    turma_order = sorted(
        cargas_por_turma.keys(),
        key=lambda t: sum(int(c.aulas_semana) for c in cargas_por_turma[t]),
        reverse=True
    )

    todas_alocacoes = []
    total_aulas = 0
    total_alocadas = 0
    turmas_falha = []

    for tid in turma_order:
        cargas = cargas_por_turma[tid]
        result = resolver_turma(tid, cargas, indisp, prof_ocupado)
        total_aulas += result["total"]

        if result["alocacoes"]:
            total_alocadas += result["alocadas"]
            todas_alocacoes.extend(result["alocacoes"])
            for aloc in result["alocacoes"]:
                di = DIAS_ACCEPT.get(aloc["dia"], -1)
                si = SLOT_TO_IDX.get(aloc["slot"], -1)
                if di >= 0 and si >= 0:
                    prof_ocupado.add((aloc["professor_id"], di, si))

        if not result["ok"]:
            turmas_falha.append(tid)

    nao_alocadas = total_aulas - total_alocadas
    turma_map = {str(t.id): t.nome for t in data.turmas}
    falha_nomes = [turma_map.get(t, t) for t in turmas_falha]

    status = "otimo" if nao_alocadas == 0 else ("viavel" if total_alocadas > 0 else "inviavel")
    msg = f"Grade gerada! {total_alocadas}/{total_aulas} aulas alocadas."
    if falha_nomes:
        msg += f" Turmas com problemas: {', '.join(falha_nomes)}"

    return {
        "status": status,
        "message": msg,
        "alocacoes": todas_alocacoes,
        "total_aulas": total_aulas,
        "alocadas": total_alocadas,
        "nao_alocadas": nao_alocadas,
        "penalidade_total": 0,
        "detalhes_solver": {
            "turmas_resolvidas": len(turma_order),
            "turmas_com_falha": len(turmas_falha)
        }
    }


def resolver_turma(turma_id, cargas, indisp, prof_ocupado):
    """
    Resolve o sub-problema de UMA turma.
    Modelo pequeno: ~N_aulas * 45 variáveis.
    """

    # Expandir cargas em aulas individuais
    aulas = []
    for c in cargas:
        n = int(c.aulas_semana)
        did = str(c.disciplina_id)
        pid = str(c.professor_id)
        gem = (c.geminada == "SIM")

        if gem and n >= 2:
            for _ in range(n // 2):
                aulas.append({"did": did, "pid": pid, "tipo": "geminada"})
            if n % 2:
                aulas.append({"did": did, "pid": pid, "tipo": "simples"})
        else:
            for _ in range(n):
                aulas.append({"did": did, "pid": pid, "tipo": "simples"})

    na = len(aulas)
    if na == 0:
        return {"ok": True, "total": 0, "alocadas": 0, "alocacoes": []}

    # Slots válidos para início de geminada
    gem_starts = set()
    for d in range(NUM_DIAS):
        for s in range(MANHA_LEN - 1):
            gem_starts.add((d, s))
        for s in range(MANHA_LEN, NUM_SLOTS - 1):
            gem_starts.add((d, s))

    # Pré-computar bloqueios por aula
    # blocked[a] = set de (d, s) onde aula a NÃO pode ser alocada
    blocked = []
    for a in range(na):
        pid = aulas[a]["pid"]
        bl = set()
        for d in range(NUM_DIAS):
            for s in range(NUM_SLOTS):
                if (pid, d, s) in indisp or (pid, d, s) in prof_ocupado:
                    bl.add((d, s))
        blocked.append(bl)

    # ── Modelo CP-SAT ──
    model = cp_model.CpModel()

    x = {}
    for a in range(na):
        for d in range(NUM_DIAS):
            for s in range(NUM_SLOTS):
                x[(a, d, s)] = model.new_bool_var(f"x{a}d{d}s{s}")

    # HC1: Cada aula alocada exatamente 1 vez
    for a in range(na):
        if aulas[a]["tipo"] == "geminada":
            valid = []
            for d in range(NUM_DIAS):
                for s in range(NUM_SLOTS):
                    if (d, s) in gem_starts:
                        # Verificar que ambos os slots não estão bloqueados
                        if (d, s) not in blocked[a] and (d, s + 1) not in blocked[a]:
                            valid.append(x[(a, d, s)])
                        else:
                            model.add(x[(a, d, s)] == 0)
                    else:
                        model.add(x[(a, d, s)] == 0)
            if not valid:
                logger.warning(f"Turma {turma_id}: aula geminada {a} sem slot viável")
                return {"ok": False, "total": na, "alocadas": 0, "alocacoes": []}
            model.add_exactly_one(valid)
        else:
            valid = []
            for d in range(NUM_DIAS):
                for s in range(NUM_SLOTS):
                    if (d, s) in blocked[a]:
                        model.add(x[(a, d, s)] == 0)
                    else:
                        valid.append(x[(a, d, s)])
            if not valid:
                logger.warning(f"Turma {turma_id}: aula {a} sem slot viável")
                return {"ok": False, "total": na, "alocadas": 0, "alocacoes": []}
            model.add_exactly_one(valid)

    # HC2: Max 1 aula por slot na turma
    for d in range(NUM_DIAS):
        for s in range(NUM_SLOTS):
            occ = []
            for a in range(na):
                occ.append(x[(a, d, s)])
                if aulas[a]["tipo"] == "geminada" and s > 0:
                    sp = s - 1
                    same = (sp < MANHA_LEN and s < MANHA_LEN) or (sp >= MANHA_LEN and s >= MANHA_LEN)
                    if same:
                        occ.append(x[(a, d, sp)])
            if len(occ) > 1:
                model.add(sum(occ) <= 1)

    # HC3: Conflito de mesmo professor dentro da turma
    prof_a = {}
    for a in range(na):
        prof_a.setdefault(aulas[a]["pid"], []).append(a)

    for pid, als in prof_a.items():
        if len(als) <= 1:
            continue
        for d in range(NUM_DIAS):
            for s in range(NUM_SLOTS):
                occ = []
                for a in als:
                    occ.append(x[(a, d, s)])
                    if aulas[a]["tipo"] == "geminada" and s > 0:
                        sp = s - 1
                        same = (sp < MANHA_LEN and s < MANHA_LEN) or (sp >= MANHA_LEN and s >= MANHA_LEN)
                        if same:
                            occ.append(x[(a, d, sp)])
                if len(occ) > 1:
                    model.add(sum(occ) <= 1)

    # ── SOFT: max 2 da mesma disciplina por dia + distribuição ──
    penalties = []
    sci = [0]
    def sn():
        sci[0] += 1
        return f"p{sci[0]}"

    disc_a = {}
    for a in range(na):
        disc_a.setdefault(aulas[a]["did"], []).append(a)

    for did, als in disc_a.items():
        for d in range(NUM_DIAS):
            dvars = [x[(a, d, s)] for a in als for s in range(NUM_SLOTS)]
            if len(dvars) > 2:
                e = model.new_int_var(0, len(dvars), sn())
                model.add(e >= sum(dvars) - 2)
                penalties.append(e * 50)
            if len(als) > 1 and dvars:
                o = model.new_int_var(0, 10, sn())
                model.add(o >= sum(dvars) - 1)
                penalties.append(o * 10)

    if penalties:
        model.minimize(sum(penalties))

    # ── Resolver ──
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15
    solver.parameters.num_workers = 1

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        logger.warning(f"Turma {turma_id}: sem solução (status={status})")
        return {"ok": False, "total": na, "alocadas": 0, "alocacoes": []}

    # Extrair
    alocacoes = []
    for a in range(na):
        for d in range(NUM_DIAS):
            found = False
            for s in range(NUM_SLOTS):
                if solver.Value(x[(a, d, s)]) == 1:
                    alocacoes.append({
                        "turma_id": turma_id,
                        "disciplina_id": aulas[a]["did"],
                        "professor_id": aulas[a]["pid"],
                        "dia": DIA_NAMES[d],
                        "slot": TODOS_SLOTS[s]
                    })
                    if aulas[a]["tipo"] == "geminada" and s + 1 < NUM_SLOTS:
                        alocacoes.append({
                            "turma_id": turma_id,
                            "disciplina_id": aulas[a]["did"],
                            "professor_id": aulas[a]["pid"],
                            "dia": DIA_NAMES[d],
                            "slot": TODOS_SLOTS[s + 1]
                        })
                    found = True
                    break
            if found:
                break

    logger.info(f"Turma {turma_id}: {len(alocacoes)} aulas alocadas")
    return {"ok": True, "total": na, "alocadas": len(alocacoes), "alocacoes": alocacoes}
