use crate::{EngineError, Result};
use candle_core::{DType, Device, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeDeviceKind {
    Cpu,
    Cuda,
    Metal,
}

pub struct RuntimeDevice {
    kind: RuntimeDeviceKind,
    device: Device,
}

impl RuntimeDevice {
    pub fn select(preference: DevicePreference) -> Result<Self> {
        match preference {
            DevicePreference::Cpu => Ok(Self::cpu()),
            DevicePreference::Cuda => Self::cuda(),
            DevicePreference::Metal => Self::metal(),
            DevicePreference::Auto => Self::cuda()
                .or_else(|_| Self::metal())
                .or_else(|_| Ok(Self::cpu())),
        }
    }

    pub fn cpu() -> Self {
        Self {
            kind: RuntimeDeviceKind::Cpu,
            device: Device::Cpu,
        }
    }

    pub fn kind(&self) -> RuntimeDeviceKind {
        self.kind
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn is_accelerated(&self) -> bool {
        !matches!(self.kind, RuntimeDeviceKind::Cpu)
    }

    /// Runs a minimal tensor operation on the selected device.
    ///
    /// This verifies that a backend can allocate and execute instead of merely
    /// reporting that its feature was compiled into the binary.
    pub fn smoke_test(&self) -> Result<()> {
        let tensor = Tensor::zeros((2, 2), DType::F32, &self.device)
            .map_err(|error| EngineError::Candle(error.to_string()))?;
        tensor
            .matmul(&tensor)
            .map_err(|error| EngineError::Candle(error.to_string()))?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda() -> Result<Self> {
        let device = Device::new_cuda(0)
            .map_err(|error| EngineError::BackendUnavailable(error.to_string()))?;
        Ok(Self {
            kind: RuntimeDeviceKind::Cuda,
            device,
        })
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda() -> Result<Self> {
        Err(EngineError::BackendUnavailable(
            "CUDA support was not compiled into this binary".to_owned(),
        ))
    }

    #[cfg(feature = "metal")]
    fn metal() -> Result<Self> {
        let device = Device::new_metal(0)
            .map_err(|error| EngineError::BackendUnavailable(error.to_string()))?;
        Ok(Self {
            kind: RuntimeDeviceKind::Metal,
            device,
        })
    }

    #[cfg(not(feature = "metal"))]
    fn metal() -> Result<Self> {
        Err(EngineError::BackendUnavailable(
            "Metal support was not compiled into this binary".to_owned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_executes_a_tensor_operation() {
        let device = RuntimeDevice::select(DevicePreference::Cpu).unwrap();
        assert_eq!(device.kind(), RuntimeDeviceKind::Cpu);
        assert!(!device.is_accelerated());
        device.smoke_test().unwrap();
    }

    #[test]
    fn auto_selection_falls_back_to_a_working_backend() {
        let device = RuntimeDevice::select(DevicePreference::Auto).unwrap();
        device.smoke_test().unwrap();
    }
}
