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
from pydantic import BaseModel
from typing import Optional
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
    id: str
    nome: str

class Turma(BaseModel):
    id: str
    nome: str
    serie: str
    turno: Optional[str] = "Integral"

class Disciplina(BaseModel):
    id: str
    nome: str
    abreviacao: Optional[str] = ""

class CargaHoraria(BaseModel):
    id: str
    turma_id: str
    disciplina_id: str
    professor_id: str
    aulas_semana: int
    geminada: Optional[str] = "NAO"  # "SIM" ou "NAO"

class Indisponibilidade(BaseModel):
    professor_id: str
    dia: str
    slot: int

class InputData(BaseModel):
    professores: list[Professor]
    turmas: list[Turma]
    disciplinas: list[Disciplina]
    carga_horaria: list[CargaHoraria]
    indisponibilidades: list[Indisponibilidade]  # apenas os bloqueios
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
    except Exception as e:
        logger.error(f"Erro no solver: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro no solver: {str(e)}")

    elapsed = round(time.time() - start_time, 2)
    resultado["tempo_segundos"] = elapsed
    logger.info(f"Resolvido em {elapsed}s — status: {resultado['status']}")
    return resultado


def resolver_horarios(data: InputData) -> dict:
    """
    Modela e resolve o problema de timetabling usando CP-SAT.
    
    Variáveis de decisão:
        x[(c, d, s)] ∈ {0, 1}  — se a carga c está alocada no dia d, slot s
    
    Hard Constraints:
        1. Carga horária fechada (cada carga usa exatamente N slots na semana)
        2. Conflito de professor (professor em no máximo 1 aula por dia/slot)
        3. Conflito de turma (turma em no máximo 1 aula por dia/slot)
        4. Disponibilidade do professor (respeitar indisponibilidades)
        5. Aulas geminadas (se marcada, devem ser 2 slots consecutivos no mesmo turno)
    
    Soft Constraints (minimizar penalidades):
        1. Limitar aulas da mesma disciplina/turma a máx 2 por dia
        2. Distribuir aulas ao longo da semana (evitar concentrar num dia)
        3. Evitar que professor dê aula todos os 5 dias
        4. Minimizar janelas dos professores
    """

    model = cp_model.CpModel()

    # ── Índices ──
    cargas = data.carga_horaria
    num_cargas = len(cargas)

    # Mapas rápidos
    prof_ids = {p.id for p in data.professores}
    turma_ids = {t.id for t in data.turmas}
    disc_ids = {d.id for d in data.disciplinas}

    # Indisponibilidades: set de (prof_id, dia_idx, slot_idx)
    indisp_set = set()
    dia_to_idx = {d: i for i, d in enumerate(DIAS)}
    slot_to_idx = {s: i for i, s in enumerate(TODOS_SLOTS)}

    for ind in data.indisponibilidades:
        d_idx = dia_to_idx.get(ind.dia)
        s_idx = slot_to_idx.get(ind.slot)
        if d_idx is not None and s_idx is not None:
            indisp_set.add((ind.professor_id, d_idx, s_idx))

    # ── Expandir cargas em "aulas" ──
    # Cada carga gera N aulas individuais (ou pares geminados)
    # aula = (carga_idx, turma_id, disc_id, prof_id, é_geminada)
    aulas = []
    aula_meta = []  # metadados de cada aula

    for c_idx, c in enumerate(cargas):
        n = c.aulas_semana
        is_gem = c.geminada == "SIM"

        if is_gem and n >= 2:
            pares = n // 2
            sobra = n % 2
            for _ in range(pares):
                aulas.append(len(aula_meta))
                aula_meta.append({
                    "carga_idx": c_idx,
                    "turma_id": c.turma_id,
                    "disc_id": c.disciplina_id,
                    "prof_id": c.professor_id,
                    "tipo": "geminada",
                    "slots_necessarios": 2
                })
            for _ in range(sobra):
                aulas.append(len(aula_meta))
                aula_meta.append({
                    "carga_idx": c_idx,
                    "turma_id": c.turma_id,
                    "disc_id": c.disciplina_id,
                    "prof_id": c.professor_id,
                    "tipo": "simples",
                    "slots_necessarios": 1
                })
        else:
            for _ in range(n):
                aulas.append(len(aula_meta))
                aula_meta.append({
                    "carga_idx": c_idx,
                    "turma_id": c.turma_id,
                    "disc_id": c.disciplina_id,
                    "prof_id": c.professor_id,
                    "tipo": "simples",
                    "slots_necessarios": 1
                })

    num_aulas = len(aula_meta)
    logger.info(f"Total de aulas a alocar: {num_aulas}")

    if num_aulas == 0:
        return {
            "status": "ok",
            "message": "Nenhuma aula para alocar.",
            "alocacoes": [],
            "total_aulas": 0,
            "alocadas": 0,
            "nao_alocadas": 0
        }

    # ── Variáveis de decisão ──
    # Para aulas simples: x[a][d][s] = 1 se aula a está no dia d, slot s
    # Para geminadas: x[a][d][s] = 1 se aula a COMEÇA no dia d, slot s (ocupa s e s+1)
    x = {}
    for a in range(num_aulas):
        for d in range(NUM_DIAS):
            for s in range(NUM_SLOTS):
                x[(a, d, s)] = model.NewBoolVar(f"x_a{a}_d{d}_s{s}")

    # ── HARD CONSTRAINT 1: Cada aula é alocada exatamente uma vez ──
    for a in range(num_aulas):
        meta = aula_meta[a]
        if meta["tipo"] == "geminada":
            # Geminada: pode começar em slots onde cabe um par consecutivo no mesmo turno
            valid_starts = []
            for d in range(NUM_DIAS):
                # Manhã: slots 0..3 (pode começar em 0,1,2,3 para ocupar até 4)
                for s in range(len(SLOTS_MANHA) - 1):
                    valid_starts.append((d, s))
                # Tarde: slots 5..7 (índices 5,6,7 para slots 6,7,8,9)
                tarde_offset = len(SLOTS_MANHA)
                for s in range(tarde_offset, tarde_offset + len(SLOTS_TARDE) - 1):
                    valid_starts.append((d, s))

            # Zerar todas as variáveis fora dos starts válidos
            for d in range(NUM_DIAS):
                for s in range(NUM_SLOTS):
                    if (d, s) not in valid_starts:
                        model.Add(x[(a, d, s)] == 0)

            # Exatamente 1 start
            model.AddExactlyOne(x[(a, d, s)] for d, s in valid_starts)
        else:
            # Simples: exatamente 1 slot
            model.AddExactlyOne(x[(a, d, s)] for d in range(NUM_DIAS) for s in range(NUM_SLOTS))

    # ── HARD CONSTRAINT 2: Conflito de professor ──
    # Agrupar aulas por professor
    prof_aulas = {}
    for a in range(num_aulas):
        pid = aula_meta[a]["prof_id"]
        if pid not in prof_aulas:
            prof_aulas[pid] = []
        prof_aulas[pid].append(a)

    for pid, aulas_list in prof_aulas.items():
        for d in range(NUM_DIAS):
            for s in range(NUM_SLOTS):
                # Todas as variáveis que "ocupam" esse slot para este professor
                ocupa = []
                for a in aulas_list:
                    meta = aula_meta[a]
                    if meta["tipo"] == "geminada":
                        # Ocupa se começa em s ou em s-1
                        ocupa.append(x[(a, d, s)])
                        if s > 0:
                            # Verificar se s-1 está no mesmo turno
                            same_turno = (
                                (s - 1 < len(SLOTS_MANHA) and s < len(SLOTS_MANHA)) or
                                (s - 1 >= len(SLOTS_MANHA) and s >= len(SLOTS_MANHA))
                            )
                            if same_turno:
                                ocupa.append(x[(a, d, s - 1)])
                    else:
                        ocupa.append(x[(a, d, s)])
                if len(ocupa) > 1:
                    model.Add(sum(ocupa) <= 1)

    # ── HARD CONSTRAINT 3: Conflito de turma ──
    turma_aulas = {}
    for a in range(num_aulas):
        tid = aula_meta[a]["turma_id"]
        if tid not in turma_aulas:
            turma_aulas[tid] = []
        turma_aulas[tid].append(a)

    for tid, aulas_list in turma_aulas.items():
        for d in range(NUM_DIAS):
            for s in range(NUM_SLOTS):
                ocupa = []
                for a in aulas_list:
                    meta = aula_meta[a]
                    if meta["tipo"] == "geminada":
                        ocupa.append(x[(a, d, s)])
                        if s > 0:
                            same_turno = (
                                (s - 1 < len(SLOTS_MANHA) and s < len(SLOTS_MANHA)) or
                                (s - 1 >= len(SLOTS_MANHA) and s >= len(SLOTS_MANHA))
                            )
                            if same_turno:
                                ocupa.append(x[(a, d, s - 1)])
                    else:
                        ocupa.append(x[(a, d, s)])
                if len(ocupa) > 1:
                    model.Add(sum(ocupa) <= 1)

    # ── HARD CONSTRAINT 4: Disponibilidade do professor ──
    for a in range(num_aulas):
        meta = aula_meta[a]
        pid = meta["prof_id"]
        for d in range(NUM_DIAS):
            for s in range(NUM_SLOTS):
                if (pid, d, s) in indisp_set:
                    model.Add(x[(a, d, s)] == 0)
                    # Para geminadas, se o slot s+1 é indisponível, não pode começar em s
                    if meta["tipo"] == "geminada" and s + 1 < NUM_SLOTS:
                        if (pid, d, s + 1) in indisp_set:
                            model.Add(x[(a, d, s)] == 0)

    # ── SOFT CONSTRAINTS (penalidades) ──
    penalties = []

    # SOFT 1: Limitar aulas da mesma disciplina/turma a máx 2 por dia
    # Agrupar por (turma_id, disc_id)
    turma_disc_aulas = {}
    for a in range(num_aulas):
        key = (aula_meta[a]["turma_id"], aula_meta[a]["disc_id"])
        if key not in turma_disc_aulas:
            turma_disc_aulas[key] = []
        turma_disc_aulas[key].append(a)

    for (tid, did), aulas_list in turma_disc_aulas.items():
        for d in range(NUM_DIAS):
            # Contar slots ocupados nesse dia por essa disciplina/turma
            day_slots = []
            for a in aulas_list:
                meta = aula_meta[a]
                for s in range(NUM_SLOTS):
                    if meta["tipo"] == "geminada":
                        day_slots.append(x[(a, d, s)])
                        # Geminada também ocupa s+1 se começa em s
                    else:
                        day_slots.append(x[(a, d, s)])

            if len(day_slots) > 2:
                # Penalizar se > 2 aulas num dia
                excess = model.NewIntVar(0, len(day_slots), f"excess_td_{tid}_{did}_d{d}")
                total_day = sum(day_slots)
                model.Add(excess >= total_day - 2)
                model.Add(excess >= 0)
                penalties.append(excess * 50)

    # SOFT 2: Distribuir aulas ao longo da semana
    for (tid, did), aulas_list in turma_disc_aulas.items():
        if len(aulas_list) <= 1:
            continue
        for d in range(NUM_DIAS):
            day_count = []
            for a in aulas_list:
                for s in range(NUM_SLOTS):
                    day_count.append(x[(a, d, s)])
            if len(day_count) > 0:
                total_day = sum(day_count)
                over2 = model.NewIntVar(0, 10, f"dist_{tid}_{did}_d{d}")
                model.Add(over2 >= total_day - 1)
                model.Add(over2 >= 0)
                penalties.append(over2 * 10)

    # SOFT 3: Evitar que professor dê aula todos os 5 dias
    for pid, aulas_list in prof_aulas.items():
        day_used = []
        for d in range(NUM_DIAS):
            has_class = model.NewBoolVar(f"prof_{pid}_day{d}")
            day_vars = []
            for a in aulas_list:
                for s in range(NUM_SLOTS):
                    day_vars.append(x[(a, d, s)])
            if day_vars:
                model.AddMaxEquality(has_class, day_vars)
                day_used.append(has_class)

        if len(day_used) == 5:
            all_five = model.NewBoolVar(f"prof_{pid}_all5")
            model.Add(sum(day_used) == 5).OnlyEnforceIf(all_five)
            model.Add(sum(day_used) < 5).OnlyEnforceIf(all_five.Not())
            penalties.append(all_five * 30)

    # SOFT 4: Minimizar janelas dos professores
    for pid, aulas_list in prof_aulas.items():
        for d in range(NUM_DIAS):
            # Para cada turno, verificar janelas
            for turno_slots in [range(0, len(SLOTS_MANHA)), range(len(SLOTS_MANHA), NUM_SLOTS)]:
                slots_list = list(turno_slots)
                if len(slots_list) < 3:
                    continue
                for i in range(1, len(slots_list) - 1):
                    s_prev = slots_list[i - 1]
                    s_curr = slots_list[i]
                    s_next = slots_list[i + 1]

                    # Janela: prev ocupado, curr vazio, next ocupado
                    prev_occ = []
                    curr_occ = []
                    next_occ = []
                    for a in aulas_list:
                        meta = aula_meta[a]
                        if meta["tipo"] == "geminada":
                            prev_occ.append(x[(a, d, s_prev)])
                            curr_occ.append(x[(a, d, s_curr)])
                            next_occ.append(x[(a, d, s_next)])
                            if s_prev > 0:
                                prev_occ.append(x[(a, d, s_prev - 1)] if (s_prev - 1) in [sl for sl in slots_list] else model.NewConstant(0))
                        else:
                            prev_occ.append(x[(a, d, s_prev)])
                            curr_occ.append(x[(a, d, s_curr)])
                            next_occ.append(x[(a, d, s_next)])

                    if prev_occ and curr_occ and next_occ:
                        has_prev = model.NewBoolVar(f"hp_{pid}_d{d}_s{s_prev}")
                        has_curr = model.NewBoolVar(f"hc_{pid}_d{d}_s{s_curr}")
                        has_next = model.NewBoolVar(f"hn_{pid}_d{d}_s{s_next}")

                        model.AddMaxEquality(has_prev, prev_occ)
                        model.AddMaxEquality(has_curr, curr_occ)
                        model.AddMaxEquality(has_next, next_occ)

                        gap = model.NewBoolVar(f"gap_{pid}_d{d}_s{s_curr}")
                        model.AddBoolAnd([has_prev, has_curr.Not(), has_next]).OnlyEnforceIf(gap)
                        model.AddBoolOr([has_prev.Not(), has_curr, has_next.Not()]).OnlyEnforceIf(gap.Not())
                        penalties.append(gap * 20)

    # ── FUNÇÃO OBJETIVO ──
    if penalties:
        model.Minimize(sum(penalties))

    # ── RESOLVER ──
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120  # máximo 2 minutos
    solver.parameters.num_workers = 4
    solver.parameters.log_search_progress = False

    logger.info(f"Iniciando solver com {num_aulas} aulas, {len(data.turmas)} turmas...")
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
            "message": "Não foi possível encontrar uma solução viável. Verifique se as restrições não são conflitantes.",
            "alocacoes": [],
            "total_aulas": num_aulas,
            "alocadas": 0,
            "nao_alocadas": num_aulas,
            "detalhes_solver": {
                "status_code": status,
                "wall_time": round(solver.WallTime(), 2)
            }
        }

    # ── EXTRAIR SOLUÇÃO ──
    alocacoes = []
    for a in range(num_aulas):
        meta = aula_meta[a]
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
                    # Se geminada, adicionar o slot seguinte
                    if meta["tipo"] == "geminada":
                        alocacoes.append({
                            "turma_id": meta["turma_id"],
                            "disciplina_id": meta["disc_id"],
                            "professor_id": meta["prof_id"],
                            "dia": DIAS[d],
                            "slot": TODOS_SLOTS[s + 1]
                        })
                    break
            else:
                continue
            break

    # ── ESTATÍSTICAS ──
    total_slots_alocados = len(alocacoes)
    penalidade_total = round(solver.ObjectiveValue(), 1) if penalties else 0

    return {
        "status": result_status,
        "message": f"Grade gerada com sucesso! {total_slots_alocados} aulas alocadas.",
        "alocacoes": alocacoes,
        "total_aulas": num_aulas,
        "alocadas": total_slots_alocados,
        "nao_alocadas": 0,
        "penalidade_total": penalidade_total,
        "detalhes_solver": {
            "status_code": status,
            "wall_time": round(solver.WallTime(), 2),
            "objective": penalidade_total
        }
    }
