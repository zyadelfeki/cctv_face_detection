"""
Edge deployment module for optimized inference on edge devices.
Supports ONNX export, TensorRT optimization, and Raspberry Pi/Jetson deployment.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for edge model deployment."""
    model_path: str
    input_shape: Tuple[int, ...]
    output_names: List[str]
    input_name: str = "input"
    precision: str = "fp16"  # fp32, fp16, int8
    device: str = "cpu"  # cpu, cuda, tensorrt
    batch_size: int = 1
    dynamic_batch: bool = False


class ONNXExporter:
    """Export PyTorch models to ONNX format for edge deployment."""
    
    def __init__(
        self,
        opset_version: int = 13,
        dynamic_axes: Optional[Dict] = None
    ):
        """
        Initialize ONNX exporter.
        
        Args:
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes for variable batch size
        """
        self.opset_version = opset_version
        self.dynamic_axes = dynamic_axes or {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    
    def export_pytorch_model(
        self,
        model,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 160, 160),
        input_names: List[str] = ["input"],
        output_names: List[str] = ["embedding"],
        simplify: bool = True
    ) -> bool:
        """
        Export PyTorch model to ONNX.
        
        Args:
            model: PyTorch model
            output_path: Path to save ONNX model
            input_shape: Model input shape
            input_names: Input tensor names
            output_names: Output tensor names
            simplify: Whether to simplify the ONNX model
            
        Returns:
            True if successful
        """
        try:
            import torch
            
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            if next(model.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            
            # Export
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=self.dynamic_axes,
                opset_version=self.opset_version,
                do_constant_folding=True
            )
            
            logger.info(f"Exported ONNX model to {output_path}")
            
            # Simplify if requested
            if simplify:
                self._simplify_onnx(output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False
    
    def _simplify_onnx(self, model_path: str) -> bool:
        """Simplify ONNX model using onnx-simplifier."""
        try:
            import onnx
            from onnxsim import simplify
            
            model = onnx.load(model_path)
            model_simplified, check = simplify(model)
            
            if check:
                onnx.save(model_simplified, model_path)
                logger.info("ONNX model simplified successfully")
                return True
            else:
                logger.warning("ONNX simplification check failed")
                return False
                
        except ImportError:
            logger.warning("onnx-simplifier not installed, skipping simplification")
            return False
        except Exception as e:
            logger.error(f"ONNX simplification failed: {e}")
            return False
    
    def validate_onnx(self, model_path: str) -> bool:
        """Validate ONNX model."""
        try:
            import onnx
            
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            logger.info("ONNX model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            return False


class TensorRTOptimizer:
    """Optimize models using TensorRT for NVIDIA GPUs."""
    
    def __init__(
        self,
        workspace_size: int = 1 << 30,  # 1GB
        precision: str = "fp16",
        max_batch_size: int = 8
    ):
        """
        Initialize TensorRT optimizer.
        
        Args:
            workspace_size: Maximum workspace size in bytes
            precision: Precision mode (fp32, fp16, int8)
            max_batch_size: Maximum batch size
        """
        self.workspace_size = workspace_size
        self.precision = precision
        self.max_batch_size = max_batch_size
        self._trt = None
    
    def _init_tensorrt(self):
        """Initialize TensorRT."""
        if self._trt is None:
            try:
                import tensorrt as trt
                self._trt = trt
                self.logger = trt.Logger(trt.Logger.WARNING)
            except ImportError:
                logger.error("TensorRT not installed")
                raise
    
    def convert_onnx_to_trt(
        self,
        onnx_path: str,
        trt_path: str,
        calibration_data: Optional[np.ndarray] = None
    ) -> bool:
        """
        Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            trt_path: Path to save TensorRT engine
            calibration_data: Data for INT8 calibration
            
        Returns:
            True if successful
        """
        self._init_tensorrt()
        trt = self._trt
        
        try:
            builder = trt.Builder(self.logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, self.logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"TRT Parse Error: {parser.get_error(error)}")
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = self.workspace_size
            
            # Set precision
            if self.precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Using FP16 precision")
            elif self.precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                if calibration_data is not None:
                    config.int8_calibrator = self._create_calibrator(calibration_data)
                logger.info("Using INT8 precision")
            
            # Build engine
            builder.max_batch_size = self.max_batch_size
            engine = builder.build_engine(network, config)
            
            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # Serialize and save
            with open(trt_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT engine saved to {trt_path}")
            return True
            
        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            return False
    
    def _create_calibrator(self, data: np.ndarray):
        """Create INT8 calibrator."""
        self._init_tensorrt()
        trt = self._trt
        
        class Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, data):
                super().__init__()
                self.data = data
                self.batch_size = 1
                self.current_idx = 0
                
                import pycuda.driver as cuda
                import pycuda.autoinit
                self.device_input = cuda.mem_alloc(data[0].nbytes)
            
            def get_batch_size(self):
                return self.batch_size
            
            def get_batch(self, names):
                if self.current_idx >= len(self.data):
                    return None
                
                import pycuda.driver as cuda
                cuda.memcpy_htod(self.device_input, self.data[self.current_idx])
                self.current_idx += 1
                return [int(self.device_input)]
            
            def read_calibration_cache(self):
                return None
            
            def write_calibration_cache(self, cache):
                pass
        
        return Calibrator(data)


class EdgeInferenceEngine:
    """
    Unified inference engine for edge deployment.
    Supports ONNX Runtime, TensorRT, and OpenVINO.
    """
    
    def __init__(
        self,
        model_path: str,
        backend: str = "onnx",  # onnx, tensorrt, openvino
        device: str = "cpu",  # cpu, cuda, npu
        num_threads: int = 4
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model file
            backend: Inference backend
            device: Device to run on
            num_threads: Number of CPU threads
        """
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self.num_threads = num_threads
        
        self.session = None
        self.input_name = None
        self.output_names = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the inference session."""
        if self.backend == "onnx":
            self._init_onnx()
        elif self.backend == "tensorrt":
            self._init_tensorrt()
        elif self.backend == "openvino":
            self._init_openvino()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _init_onnx(self):
        """Initialize ONNX Runtime session."""
        try:
            import onnxruntime as ort
            
            # Configure session options
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.num_threads
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Select providers based on device
            if self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif self.device == "tensorrt":
                providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(
                self.model_path,
                options,
                providers=providers
            )
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            logger.info(f"ONNX Runtime initialized with {providers[0]}")
            
        except Exception as e:
            logger.error(f"ONNX Runtime initialization failed: {e}")
            raise
    
    def _init_tensorrt(self):
        """Initialize TensorRT engine."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # Load engine
            with open(self.model_path, 'rb') as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            
            # Allocate buffers
            self.inputs = []
            self.outputs = []
            self.bindings = []
            
            for binding in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.bindings.append(int(device_mem))
                
                if self.engine.binding_is_input(binding):
                    self.inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem})
            
            self.stream = cuda.Stream()
            
            logger.info("TensorRT engine initialized")
            
        except Exception as e:
            logger.error(f"TensorRT initialization failed: {e}")
            raise
    
    def _init_openvino(self):
        """Initialize OpenVINO for Intel devices."""
        try:
            from openvino.runtime import Core
            
            core = Core()
            
            # Load model
            model = core.read_model(self.model_path)
            
            # Compile for device
            device_map = {
                "cpu": "CPU",
                "gpu": "GPU",
                "npu": "NPU",
                "myriad": "MYRIAD"
            }
            
            target = device_map.get(self.device, "CPU")
            self.session = core.compile_model(model, target)
            
            self.input_name = model.inputs[0].get_any_name()
            self.output_names = [o.get_any_name() for o in model.outputs]
            
            logger.info(f"OpenVINO initialized with {target}")
            
        except Exception as e:
            logger.error(f"OpenVINO initialization failed: {e}")
            raise
    
    def infer(self, input_data: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Run inference on input data.
        
        Args:
            input_data: Input tensor (NCHW format)
            
        Returns:
            Model output(s)
        """
        if self.backend == "onnx":
            return self._infer_onnx(input_data)
        elif self.backend == "tensorrt":
            return self._infer_tensorrt(input_data)
        elif self.backend == "openvino":
            return self._infer_openvino(input_data)
    
    def _infer_onnx(self, input_data: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_data.astype(np.float32)}
        )
        return outputs[0] if len(outputs) == 1 else outputs
    
    def _infer_tensorrt(self, input_data: np.ndarray) -> np.ndarray:
        """Run TensorRT inference."""
        import pycuda.driver as cuda
        
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Copy output to host
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        
        self.stream.synchronize()
        
        return self.outputs[0]['host'].copy()
    
    def _infer_openvino(self, input_data: np.ndarray) -> np.ndarray:
        """Run OpenVINO inference."""
        result = self.session({self.input_name: input_data})
        output_key = list(result.keys())[0]
        return result[output_key]
    
    def benchmark(
        self,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict:
        """
        Benchmark inference performance.
        
        Args:
            input_shape: Shape of input tensor
            num_iterations: Number of inference runs
            warmup: Number of warmup runs
            
        Returns:
            Benchmark results
        """
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup):
            self.infer(dummy_input)
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.infer(dummy_input)
            latencies.append((time.perf_counter() - start) * 1000)  # ms
        
        latencies = np.array(latencies)
        
        return {
            "backend": self.backend,
            "device": self.device,
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "throughput_fps": float(1000 / np.mean(latencies))
        }


class JetsonOptimizer:
    """Specific optimizations for NVIDIA Jetson devices."""
    
    @staticmethod
    def set_power_mode(mode: str = "MAXN"):
        """
        Set Jetson power mode.
        
        Args:
            mode: Power mode (MAXN, 15W, 10W, etc.)
        """
        import subprocess
        
        mode_map = {
            "MAXN": "0",
            "15W": "1",
            "10W": "2"
        }
        
        if mode in mode_map:
            try:
                subprocess.run(
                    ["sudo", "nvpmodel", "-m", mode_map[mode]],
                    check=True
                )
                logger.info(f"Set Jetson power mode to {mode}")
            except Exception as e:
                logger.error(f"Failed to set power mode: {e}")
    
    @staticmethod
    def set_max_clocks():
        """Set Jetson to maximum clock speeds."""
        import subprocess
        
        try:
            subprocess.run(
                ["sudo", "jetson_clocks"],
                check=True
            )
            logger.info("Set Jetson to maximum clocks")
        except Exception as e:
            logger.error(f"Failed to set max clocks: {e}")
    
    @staticmethod
    def get_system_info() -> Dict:
        """Get Jetson system information."""
        info = {
            "device": "Unknown",
            "cuda_version": "Unknown",
            "tensorrt_version": "Unknown"
        }
        
        try:
            # Get device info
            with open("/proc/device-tree/model", "r") as f:
                info["device"] = f.read().strip()
        except:
            pass
        
        try:
            import torch
            info["cuda_version"] = torch.version.cuda
        except:
            pass
        
        try:
            import tensorrt
            info["tensorrt_version"] = tensorrt.__version__
        except:
            pass
        
        return info


class RaspberryPiOptimizer:
    """Specific optimizations for Raspberry Pi deployment."""
    
    @staticmethod
    def get_system_info() -> Dict:
        """Get Raspberry Pi system information."""
        info = {
            "model": "Unknown",
            "memory_mb": 0,
            "cpu_count": 0
        }
        
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "Model" in line:
                        info["model"] = line.split(":")[1].strip()
                        break
        except:
            pass
        
        try:
            import psutil
            info["memory_mb"] = psutil.virtual_memory().total // (1024 * 1024)
            info["cpu_count"] = psutil.cpu_count()
        except:
            pass
        
        return info
    
    @staticmethod
    def optimize_for_inference():
        """Apply Raspberry Pi specific optimizations."""
        import os
        
        # Set thread affinity for better cache usage
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        
        # Disable GPU (not available on Pi)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        logger.info("Applied Raspberry Pi optimizations")
    
    @staticmethod
    def quantize_model_for_pi(
        model_path: str,
        output_path: str,
        calibration_data: Optional[np.ndarray] = None
    ) -> bool:
        """
        Quantize model for Raspberry Pi deployment.
        Uses TFLite for best Pi performance.
        """
        try:
            import tensorflow as tf
            
            # Convert ONNX to TFLite via TF SavedModel
            # This is a simplified example
            
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if calibration_data is not None:
                def representative_dataset():
                    for data in calibration_data:
                        yield [data.astype(np.float32)]
                
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            
            tflite_model = converter.convert()
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Quantized model saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return False
