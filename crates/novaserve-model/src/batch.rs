use crate::error::ModelError;
use novaserve_kv::BlockId;

/// Unified heterogeneous batch forward descriptor (`BatchInput`) combining single-token
/// `Decode` entries, `ChunkedPrefill` slices, and `K+1` `Speculative` verification tokens.
#[derive(Clone, Debug, Default)]
pub struct BatchInput {
    pub token_ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub seq_slot_idx: Vec<usize>,
    pub kv_write_block_id: Vec<BlockId>,
    pub kv_write_block_offset: Vec<usize>,
    pub is_logit_output: Vec<bool>,
}

impl BatchInput {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            token_ids: Vec::with_capacity(cap),
            positions: Vec::with_capacity(cap),
            seq_slot_idx: Vec::with_capacity(cap),
            kv_write_block_id: Vec::with_capacity(cap),
            kv_write_block_offset: Vec::with_capacity(cap),
            is_logit_output: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn push_token(
        &mut self,
        token_id: u32,
        position: u32,
        seq_slot: usize,
        write_block: BlockId,
        write_offset: usize,
        emit_logits: bool,
    ) {
        self.token_ids.push(token_id);
        self.positions.push(position);
        self.seq_slot_idx.push(seq_slot);
        self.kv_write_block_id.push(write_block);
        self.kv_write_block_offset.push(write_offset);
        self.is_logit_output.push(emit_logits);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    #[inline]
    pub fn num_logit_outputs(&self) -> usize {
        self.is_logit_output.iter().filter(|&&b| b).count()
    }

    pub fn validate(&self, num_seq_slots: usize, vocab_size: usize) -> Result<(), ModelError> {
        let n = self.token_ids.len();
        if n == 0 {
            return Err(ModelError::InvalidBatch("empty BatchInput".into()));
        }
        if self.positions.len() != n
            || self.seq_slot_idx.len() != n
            || self.kv_write_block_id.len() != n
            || self.kv_write_block_offset.len() != n
            || self.is_logit_output.len() != n
        {
            return Err(ModelError::InvalidBatch(
                "BatchInput parallel array length mismatch".into(),
            ));
        }
        for i in 0..n {
            if (self.token_ids[i] as usize) >= vocab_size {
                return Err(ModelError::InvalidBatch(format!(
                    "token_id {} >= vocab_size {vocab_size}",
                    self.token_ids[i]
                )));
            }
            if self.seq_slot_idx[i] >= num_seq_slots {
                return Err(ModelError::InvalidBatch(format!(
                    "seq_slot_idx {} >= num_seq_slots {num_seq_slots}",
                    self.seq_slot_idx[i]
                )));
            }
        }
        Ok(())
    }
}
