import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
from ultralytics import YOLO
from PIL import Image
import io
import base64
import yaml
from datetime import datetime
from roboflow import Roboflow
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


# Constante para la ruta del dataset del proyecto
DEFAULT_DATA_YAML = "/home/terrazalt/Documents/magister/Repetitive-Archetypes-Patterns-RL-APP/CNN/Repetitive-Patterns-MTI-2/data.yaml"


class RLHFTrainer:
    """
    Clase principal para manejar el entrenamiento con Human-in-the-Loop Feedback (RLHF)
    """

    # Constante para el dataset del proyecto
    DEFAULT_DATASET_PATH = "/home/terrazalt/Documents/magister/Repetitive-Archetypes-Patterns-RL-APP/CNN/Repetitive-Patterns-MTI-2/data.yaml"

    def __init__(self, model_path: str = None, images_dir: str = None):
        """
        Inicializa el entrenador RLHF

        Args:
            model_path: Ruta al modelo YOLO (por defecto usa yolo11n.pt)
            images_dir: Directorio con imágenes para feedback (por defecto src/images/)
        """
        # Configuración de rutas
        self.project_root = Path(__file__).parent.parent.parent.parent  # CNN/
        self.model_path = model_path or str(
            self.project_root / "best_models" / "best.pt"
        )
        self.images_dir = images_dir or str(self.project_root / "src" / "images")
        self.rlhf_config_path = (
            self.project_root / "src" / "utils" / "RLHF" / "rlhf_config.json"
        )

        # Inicializar modelo
        self.model = None
        self.current_epoch = 0

        print(f"Inicializando RLHFTrainer:")
        print(f"  - Modelo: {self.model_path}")
        print(f"  - Imágenes: {self.images_dir}")
        print(f"  - Config RLHF: {self.rlhf_config_path}")

    def prepare_roboflow_dataset(self) -> str:
        """
        Prepara el dataset desde Roboflow y realiza backup del modelo

        Returns:
            Ruta al archivo data.yaml del dataset preparado
        """
        try:
            print("📡 Preparando dataset desde Roboflow...")

            # Paso 1: Backup del modelo actual
            backup_path = (
                f"best_models/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            )
            shutil.copy(self.model_path, backup_path)
            print(f"💾 Backup del modelo creado en: {backup_path}")

            # Paso 2: Conectar a Roboflow
            rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
            project = rf.workspace(os.getenv("WORKSPACE_ID")).project(
                os.getenv("PROJECT_ID")
            )
            version = project.version(2)

            print("🔗 Conectado a Roboflow, descargando dataset...")
            dataset = version.download("yolov8")

            # Paso 3: Usar el data.yaml local configurado
            data_yaml_path = self.DEFAULT_DATASET_PATH
            print(f"📊 Dataset preparado. Usando data.yaml: {data_yaml_path}")

            return data_yaml_path

        except Exception as e:
            print(f"❌ Error preparando dataset Roboflow: {e}")
            raise e

    def load_model(self) -> YOLO:
        """
        Carga el modelo YOLO desde la ruta especificada

        Returns:
            Modelo YOLO cargado
        """
        try:
            print(f"Cargando modelo desde: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("✅ Modelo cargado exitosamente")
            return self.model
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            raise e

    def initial_training(
        self, epochs: int = 3, data_yaml: str | None = None, use_roboflow: bool = True
    ) -> YOLO:
        """
        Realiza un entrenamiento inicial del modelo por pocas épocas

        Args:
            epochs: Número de épocas para el entrenamiento inicial (por defecto 3)
            data_yaml: Ruta al archivo YAML de configuración de datos
            use_roboflow: Si usar Roboflow para preparar el dataset (por defecto True)

        Returns:
            Modelo entrenado
        """
        if self.model is None:
            raise ValueError("Modelo no cargado. Ejecuta load_model() primero.")

        try:
            print(f"🚀 Iniciando entrenamiento inicial por {epochs} épocas...")

            # Si no se proporciona data_yaml, preparar dataset
            if data_yaml is None:
                if use_roboflow:
                    # Preparar dataset desde Roboflow
                    data_yaml = self.prepare_roboflow_dataset()
                else:
                    # Usar el dataset real disponible en el proyecto
                    data_yaml = self.DEFAULT_DATASET_PATH
                    print(
                        f"⚠️  No se especificó data_yaml. Usando dataset del proyecto: {data_yaml}"
                    )

            # Entrenar el modelo con parámetros específicos para Roboflow
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=640,
                batch=16,
                patience=50,
                save=True,
                verbose=True,
            )

            self.current_epoch += epochs
            print(
                f"✅ Entrenamiento inicial completado. Épocas totales: {self.current_epoch}"
            )

            return self.model

        except Exception as e:
            print(f"❌ Error durante el entrenamiento inicial: {e}")
            raise e

    def get_predictions_for_feedback(self, num_images: int = 4) -> List[Dict]:
        """
        Obtiene predicciones del modelo actual para solicitar feedback humano

        Args:
            num_images: Número de imágenes para obtener predicciones (por defecto 4)

        Returns:
            Lista de diccionarios con imagen_base64, predicciones y metadatos
        """
        if self.model is None:
            raise ValueError("Modelo no cargado. Ejecuta load_model() primero.")

        try:
            print(f"📷 Obteniendo predicciones para {num_images} imágenes...")

            # Obtener lista de imágenes disponibles
            image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            image_files = [
                os.path.join(self.images_dir, f)
                for f in os.listdir(self.images_dir)
                if f.lower().endswith(image_extensions)
            ]

            if len(image_files) == 0:
                raise ValueError(f"No se encontraron imágenes en {self.images_dir}")

            # Seleccionar las primeras N imágenes
            selected_images = image_files[: min(num_images, len(image_files))]
            print(
                f"📁 Imágenes seleccionadas: {[os.path.basename(img) for img in selected_images]}"
            )

            # Hacer predicciones
            results = self.model.predict(selected_images, save=False, verbose=False)

            predictions_data = []
            for i, (img_path, result) in enumerate(zip(selected_images, results)):
                # Obtener imagen con predicciones dibujadas
                annotated_img = result.plot()  # Imagen con bounding boxes

                # Convertir de BGR (OpenCV) a RGB (PIL)
                annotated_img_rgb = annotated_img[:, :, ::-1]
                pil_img = Image.fromarray(annotated_img_rgb)

                # Convertir a base64
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="PNG")
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

                # Extraer información de las predicciones
                boxes_info = []
                if result.boxes is not None:
                    for box in result.boxes:
                        boxes_info.append(
                            {
                                "confidence": float(box.conf.cpu().numpy()[0]),
                                "class_id": int(box.cls.cpu().numpy()[0]),
                                "class_name": result.names[
                                    int(box.cls.cpu().numpy()[0])
                                ],
                                "bbox": box.xywh.cpu()
                                .numpy()[0]
                                .tolist(),  # [x_center, y_center, width, height]
                            }
                        )

                prediction_item = {
                    "image_id": i,
                    "image_name": os.path.basename(img_path),
                    "image_base64": img_base64,
                    "predictions": boxes_info,
                    "total_detections": len(boxes_info),
                    "epoch": self.current_epoch,
                }

                predictions_data.append(prediction_item)

            print(f"✅ Predicciones generadas para {len(predictions_data)} imágenes")
            return predictions_data

        except Exception as e:
            print(f"❌ Error al obtener predicciones: {e}")
            raise e

    def process_human_feedback(self, feedback_data: Dict) -> Dict:
        """
        Procesa el feedback humano y crea/actualiza el archivo rlhf_config.json

        Args:
            feedback_data: Diccionario con el feedback del humano
            Formato esperado: {
                "overall_quality": 0.7,  # 0.0 = muy malo, 1.0 = excelente
                "comments": "Las predicciones son buenas en general"
            }

        Returns:
            Configuración RLHF actualizada
        """
        try:
            print("🔄 Procesando feedback humano...")

            # Obtener calidad general de las predicciones
            overall_quality = feedback_data.get("overall_quality", 0.5)
            comments = feedback_data.get("comments", "Sin comentarios")

            # Calcular reward_factor basado en la calidad:
            # LÓGICA CORRECTA: Menor loss = Mejor predicción
            # - Si calidad > 0.7: factor negativo (disminuye loss) → MEJORA
            # - Si calidad < 0.3: factor positivo (aumenta loss) → EMPEORA
            # - Si calidad 0.3-0.7: factor pequeño

            if overall_quality >= 0.7:
                # Predicciones buenas: DISMINUIR loss (factor negativo)
                reward_factor = -0.1 * (
                    overall_quality - 0.5
                )  # -0.02 a -0.05 (negativo = menos loss)
                interpretation = (
                    "Buenas predicciones - Factor negativo (DISMINUYE loss general)"
                )
            elif overall_quality <= 0.3:
                # Predicciones malas: AUMENTAR loss (factor positivo)
                reward_factor = 0.1 * (
                    0.5 - overall_quality
                )  # +0.02 a +0.02 (positivo = más loss)
                interpretation = (
                    "Malas predicciones - Factor positivo (AUMENTA loss general)"
                )
            else:
                # Predicciones regulares: ajuste mínimo
                reward_factor = 0.05 * (overall_quality - 0.5)  # -0.01 a +0.01
                interpretation = "Predicciones regulares - Ajuste mínimo"

            # Crear configuración RLHF CORREGIDA - sin reward_cls
            rlhf_config = {
                "enable": True,
                "epoch_trigger": self.current_epoch,  # Aplicar desde la época actual
                "reward_factor": round(reward_factor, 3),
                "feedback_quality": overall_quality,  # Para tracking
            }

            # Agregar metadatos para tracking (opcional)
            rlhf_config["_metadata"] = {
                "timestamp": str(datetime.now()),
                "epoch_when_feedback_given": self.current_epoch,
                "original_quality_score": overall_quality,
                "human_comments": comments,
                "interpretation": interpretation,
            }

            # Guardar configuración RLHF
            os.makedirs(os.path.dirname(self.rlhf_config_path), exist_ok=True)
            with open(self.rlhf_config_path, "w") as f:
                json.dump(rlhf_config, f, indent=2)

            print(f"✅ Configuración RLHF guardada en: {self.rlhf_config_path}")
            print(f"   - Calidad general: {overall_quality:.3f}")
            print(f"   - Reward factor: {reward_factor:.3f}")
            print(f"   - Efecto: {interpretation}")

            return rlhf_config

        except Exception as e:
            print(f"❌ Error al procesar feedback: {e}")
            raise e

    def continue_training_with_feedback(
        self, additional_epochs: int = 5, data_yaml: str | None = None
    ) -> YOLO:
        """
        Continúa el entrenamiento del modelo utilizando el feedback humano ya procesado

        Args:
            additional_epochs: Épocas adicionales de entrenamiento con feedback
            data_yaml: Ruta al archivo YAML de configuración de datos

        Returns:
            Modelo entrenado con feedback
        """
        if self.model is None:
            raise ValueError("Modelo no cargado. Ejecuta load_model() primero.")

        # Verificar que existe configuración RLHF
        if not os.path.exists(self.rlhf_config_path):
            raise ValueError(
                "No existe configuración RLHF. Procesa feedback humano primero."
            )

        try:
            print(
                f"🚀 Continuando entrenamiento con feedback por {additional_epochs} épocas..."
            )

            # Cargar configuración RLHF para verificar
            with open(self.rlhf_config_path, "r") as f:
                rlhf_config = json.load(f)

            print(f"📊 Aplicando reward_factor: {rlhf_config['reward_factor']}")
            print(f"📊 Calidad feedback: {rlhf_config.get('feedback_quality', 'N/A')}")
            print(f"📊 Desde época: {rlhf_config['epoch_trigger']}")

            # Usar el dataset configurado si no se especifica
            if data_yaml is None:
                data_yaml = self.DEFAULT_DATASET_PATH
                print(f"⚠️  Usando dataset configurado: {data_yaml}")

            # Continuar entrenamiento con parámetros optimizados para Roboflow
            # El reward se aplicará automáticamente a través de la función de loss modificada
            self.model.train(
                data=data_yaml,
                epochs=additional_epochs,
                imgsz=640,
                batch=16,
                patience=50,
                save=True,
                verbose=True,
                resume=False,  # No resume, continúa desde el estado actual
            )

            self.current_epoch += additional_epochs
            print("✅ Entrenamiento con feedback completado.")
            print(f"📈 Épocas totales: {self.current_epoch}")

            return self.model

        except Exception as e:
            print(f"❌ Error durante entrenamiento con feedback: {e}")
            raise e

    def full_rlhf_cycle(
        self,
        initial_epochs: int = 3,
        feedback_epochs: int = 5,
        data_yaml: str | None = None,
        num_images: int = 4,
        use_roboflow: bool = True,
    ) -> Dict:
        """
        Ejecuta el ciclo completo de RLHF:
        1. Entrenamiento inicial con dataset Roboflow
        2. Obtener predicciones para feedback
        3. (Espera feedback humano - se procesará externamente)
        4. Continuar entrenamiento con feedback

        Args:
            initial_epochs: Épocas de entrenamiento inicial
            feedback_epochs: Épocas adicionales con feedback
            data_yaml: Ruta al archivo de configuración de datos
            num_images: Número de imágenes para feedback
            use_roboflow: Si usar Roboflow para preparar el dataset (por defecto True)

        Returns:
            Diccionario con el estado del proceso y predicciones para feedback
        """
        try:
            print("🔄 Iniciando ciclo completo RLHF con Roboflow...")

            # Paso 1: Entrenamiento inicial con Roboflow
            self.initial_training(
                epochs=initial_epochs, data_yaml=data_yaml, use_roboflow=use_roboflow
            )

            # Paso 2: Obtener predicciones para feedback humano
            predictions = self.get_predictions_for_feedback(num_images=num_images)

            return {
                "status": "waiting_for_feedback",
                "message": "Entrenamiento inicial completado. Esperando feedback humano.",
                "predictions_for_feedback": predictions,
                "current_epoch": self.current_epoch,
                "next_step": "Procesar feedback y continuar entrenamiento",
            }

        except Exception as e:
            print(f"❌ Error en ciclo RLHF: {e}")
            raise e


if __name__ == "__main__":
    # Ejemplo de uso completo del sistema RLHF
    print("🚀 Iniciando ejemplo de RLHF con YOLO...")

    # Inicializar trainer
    trainer = RLHFTrainer()

    # Cargar modelo
    model = trainer.load_model()

    # Ejecutar ciclo inicial (hasta obtener predicciones para feedback)
    result = trainer.full_rlhf_cycle(
        initial_epochs=2,  # Pocas épocas para la demo
        feedback_epochs=3,  # Épocas adicionales con feedback
        num_images=4,  # 4 imágenes para evaluar
    )

    print(f"\n📊 Estado: {result['status']}")
    print(f"📊 Mensaje: {result['message']}")
    print(f"📊 Época actual: {result['current_epoch']}")
    print(f"📊 Imágenes para feedback: {len(result['predictions_for_feedback'])}")

    # En un escenario real, aquí se enviarían las predicciones al frontend
    # y se esperaría el feedback del usuario

    print("\n🔄 Para completar el ciclo:")
    print("1. Envía 'predictions_for_feedback' al frontend")
    print("2. Recibe feedback del usuario")
    print("3. Llama a trainer.process_human_feedback(feedback_data)")
    print("4. Llama a trainer.continue_training_with_feedback()")

    # Ejemplo de cómo se procesaría el feedback:
    print("\n📝 Ejemplo de feedback que esperamos del frontend:")
    example_feedback = {
        "overall_quality": 0.75,  # 0.0 = muy malo, 1.0 = excelente
        "comments": "Las predicciones son bastante buenas, algunas detecciones correctas",
    }
    print(json.dumps(example_feedback, indent=2))

    print("\n🔍 Interpretación del feedback (LÓGICA CORRECTA):")
    print(
        "- overall_quality >= 0.7: Buenas predicciones → reward_factor negativo (DISMINUYE loss = MEJORA)"
    )
    print(
        "- overall_quality <= 0.3: Malas predicciones → reward_factor positivo (AUMENTA loss = EMPEORA)"
    )
    print("- overall_quality 0.3-0.7: Predicciones regulares → reward_factor mínimo")
