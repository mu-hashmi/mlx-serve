use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::ServerError;

/// MLX runtime execution is not safe under parallel decode in this server yet.
///
/// This means user-facing admission controls do not increase decode parallelism.
const MAX_PARALLEL_GENERATIONS: usize = 1;

/// Bounded queue + bounded concurrency controller.
#[derive(Clone)]
pub struct BackpressureController {
    queue_slots: Arc<Semaphore>,
    run_slots: Arc<Semaphore>,
    retry_after_seconds: u64,
}

/// A queue slot acquired for a request.
pub struct QueueTicket {
    queue_permit: OwnedSemaphorePermit,
    run_slots: Arc<Semaphore>,
}

/// A running request permit.
pub struct RunPermit {
    _queue_permit: OwnedSemaphorePermit,
    _run_permit: OwnedSemaphorePermit,
}

impl BackpressureController {
    /// Build a controller from admission and queue capacities.
    pub fn new(
        max_admitted_requests: usize,
        max_queue_size: usize,
        retry_after_seconds: u64,
    ) -> Self {
        let admitted = max_admitted_requests.max(1);
        let running = admitted.clamp(1, MAX_PARALLEL_GENERATIONS);
        let total_slots = admitted.saturating_add(max_queue_size).max(1);
        Self {
            queue_slots: Arc::new(Semaphore::new(total_slots)),
            run_slots: Arc::new(Semaphore::new(running)),
            retry_after_seconds,
        }
    }

    /// Try to queue a request immediately.
    pub fn try_queue(&self) -> Result<QueueTicket, ServerError> {
        match self.queue_slots.clone().try_acquire_owned() {
            Ok(queue_permit) => Ok(QueueTicket {
                queue_permit,
                run_slots: Arc::clone(&self.run_slots),
            }),
            Err(_) => Err(ServerError::Overloaded {
                retry_after_seconds: self.retry_after_seconds,
            }),
        }
    }
}

impl QueueTicket {
    /// Wait until the request can begin generation.
    pub async fn wait_for_run(self) -> Result<RunPermit, ServerError> {
        let run_permit = self.run_slots.acquire_owned().await.map_err(|error| {
            ServerError::InternalError(format!("backpressure run semaphore closed: {error}"))
        })?;
        Ok(RunPermit {
            _queue_permit: self.queue_permit,
            _run_permit: run_permit,
        })
    }
}
