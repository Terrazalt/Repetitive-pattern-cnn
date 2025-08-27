from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Agregar el path para importar nuestro RLHFTrainer
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.RLHF.main import RLHFTrainer

# Crear el router
router = APIRouter()

# Variable global para mantener el trainer (en producción usar Redis/DB)
trainer_instance: Optional[RLHFTrainer] = None

# ===== MODELOS PYDANTIC =====


class TrainingStartRequest(BaseModel):
    initial_epochs: int = 3
    data_yaml: Optional[str] = None
    num_images: int = 4


class FeedbackRequest(BaseModel):
    overall_quality: float  # 0.0 = muy malo, 1.0 = excelente
    comments: str = "Sin comentarios"


class ContinueTrainingRequest(BaseModel):
    additional_epochs: int = 5
    data_yaml: Optional[str] = None


# ===== MODELOS DE RESPUESTA =====


class TrainingStartResponse(BaseModel):
    status: str
    message: str
    predictions_for_feedback: List[Dict[str, Any]]
    current_epoch: int
    next_step: str


class FeedbackResponse(BaseModel):
    status: str
    message: str
    rlhf_config: Dict[str, Any]
    reward_factor: float
    feedback_quality: Optional[float] = None


class ContinueTrainingResponse(BaseModel):
    status: str
    message: str
    current_epoch: int
    final_model_path: str


# ===== ENDPOINT 1: INICIAR ENTRENAMIENTO RLHF =====


@router.post("/start-training", response_model=TrainingStartResponse)
async def start_rlhf_training(request: TrainingStartRequest):
    """
    Inicia el ciclo RLHF:
    1. Carga el modelo
    2. Entrena por pocas épocas
    3. Genera predicciones para feedback humano
    """
    global trainer_instance

    try:
        print(f"🚀 Iniciando entrenamiento RLHF...")

        # Crear nueva instancia del trainer
        trainer_instance = RLHFTrainer()

        # Cargar modelo
        trainer_instance.load_model()

        # Ejecutar ciclo inicial hasta obtener predicciones
        result = trainer_instance.full_rlhf_cycle(
            initial_epochs=request.initial_epochs,
            data_yaml=request.data_yaml,
            num_images=request.num_images,
        )

        return TrainingStartResponse(
            status=result["status"],
            message=result["message"],
            predictions_for_feedback=result["predictions_for_feedback"],
            current_epoch=result["current_epoch"],
            next_step=result["next_step"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al iniciar entrenamiento: {str(e)}"
        )


# ===== ENDPOINT 2: PROCESAR FEEDBACK HUMANO =====


@router.post("/submit-feedback", response_model=FeedbackResponse)
async def submit_human_feedback(request: FeedbackRequest):
    """
    Procesa el feedback humano y genera la configuración RLHF:
    1. Recibe la calidad general (0.0 - 1.0)
    2. Calcula reward_factor basado en calidad
    3. Genera rlhf_config.json
    4. Retorna la configuración aplicada
    """
    global trainer_instance

    # Verificar que existe una instancia de entrenamiento activa
    if trainer_instance is None:
        raise HTTPException(
            status_code=400,
            detail="No hay entrenamiento activo. Ejecuta /start-training primero.",
        )

    try:
        print("🔄 Procesando feedback humano...")

        # Preparar data del feedback
        feedback_data = {
            "overall_quality": request.overall_quality,
            "comments": request.comments,
        }

        # Procesar feedback y generar rlhf_config.json
        rlhf_config = trainer_instance.process_human_feedback(feedback_data)

        return FeedbackResponse(
            status="feedback_processed",
            message=f"Feedback procesado. Reward calculado: {rlhf_config['reward_factor']:.3f}",
            rlhf_config=rlhf_config,
            reward_factor=rlhf_config["reward_factor"],
            feedback_quality=rlhf_config.get("feedback_quality"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al procesar feedback: {str(e)}"
        )


# ===== ENDPOINT 3: COMPLETAR ENTRENAMIENTO CON FEEDBACK =====


@router.post("/complete-training", response_model=ContinueTrainingResponse)
async def complete_training_with_feedback(request: ContinueTrainingRequest):
    """
    Completa el entrenamiento aplicando el feedback humano:
    1. Verifica que el feedback ya fue procesado
    2. Continúa el entrenamiento con reward/penalty aplicado
    3. Guarda el modelo final
    4. Retorna la información del modelo completado
    """
    global trainer_instance

    # Verificar que existe una instancia de entrenamiento activa
    if trainer_instance is None:
        raise HTTPException(
            status_code=400,
            detail="No hay entrenamiento activo. Ejecuta /start-training primero.",
        )

    try:
        print("🚀 Completando entrenamiento con feedback aplicado...")

        # Continuar entrenamiento con el feedback ya procesado
        # El reward/penalty se aplicará automáticamente a través de rlhf_config.json
        final_model = trainer_instance.continue_training_with_feedback(
            additional_epochs=request.additional_epochs, data_yaml=request.data_yaml
        )

        # Obtener ruta del modelo final
        final_model_path = trainer_instance.model_path

        return ContinueTrainingResponse(
            status="training_completed",
            message=f"Entrenamiento RLHF completado exitosamente con {trainer_instance.current_epoch} épocas totales.",
            current_epoch=trainer_instance.current_epoch,
            final_model_path=final_model_path,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al completar entrenamiento: {str(e)}"
        )


# ===== ENDPOINT EXTRA: STATUS DEL ENTRENAMIENTO =====


@router.get("/status")
async def get_training_status():
    """
    Obtiene el estado actual del entrenamiento RLHF
    """
    global trainer_instance

    if trainer_instance is None:
        return {
            "status": "no_training",
            "message": "No hay entrenamiento activo",
            "current_epoch": 0,
            "model_loaded": False,
        }

    # Verificar si existe configuración RLHF
    rlhf_config_exists = trainer_instance.rlhf_config_path.exists()

    return {
        "status": "training_active",
        "message": "Entrenamiento RLHF en progreso",
        "current_epoch": trainer_instance.current_epoch,
        "model_loaded": trainer_instance.model is not None,
        "model_path": trainer_instance.model_path,
        "images_dir": trainer_instance.images_dir,
        "rlhf_config_exists": rlhf_config_exists,
        "next_step": "submit-feedback"
        if not rlhf_config_exists
        else "complete-training",
    }


if __name__ == "__main__":
    print("Router RLHF creado. Endpoints disponibles:")
    print("- POST /start-training: Inicia el entrenamiento RLHF")
    print("- POST /submit-feedback: Procesa feedback humano")
    print("- POST /complete-training: Completa entrenamiento con feedback")
    print("- GET  /status: Estado actual del entrenamiento")
