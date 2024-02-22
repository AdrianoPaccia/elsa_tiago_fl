
class BufferProcess(mp.Process):
    """
    This process takes care of the memory. It retreives trajectories from the worker queques and samples batches
    of trajctories for the training.
    """
    def __init__(self, memory, batch_queue, replay_queues, config ,termination_event,screen=False):
        super(BufferProcess, self).__init__()
        self.replay_buffer = memory
        self.replay_queues = replay_queues
        self.batch_queue = batch_queue
        self.termination_event = termination_event
        self.config = config
        self.screen = screen

    def run(self):
        log_debug('Buffer process ready to store experiences!',self.screen)

        tot_step = 1
        while not self.termination_event.is_set():
            # get new experience (if available)
            self.get_transitions()

            # send replanish the batches for training
            queue_capacity = self.batch_queue.size()/self.batch_queue.maxsize
            if len(self.replay_buffer.memory) > self.config.min_len_replay_buffer:
                if queue_capacity<0.2:
                    logger.debug(f'replanishing beacuse queue_capacity = {queue_capacity}')
                    self.batch_queue.replenish(self.replay_buffer,self.config.batch_size)
            tot_step +=1


    def get_transitions(self):
        try:
            for queue in self.replay_queues:
                experiences = []
                if not queue.empty():
                    experiences = queue.rollout()
                for exp in experiences:
                    self.replay_buffer.push(exp) 
                #while not queue.empty():
                #    exp = queue.get()
                #    self.replay_buffer.push(exp) 
                #    log_debug(f'len queue {queue.qsize()}',True)

        except Exception as e:
            logger.debug(f"Error getting shared params: {e}")
            return None


class BatchQueue:
    def __init__(self,queue,lock):
        self.queue = queue
        self.lock = lock
        self.maxsize = self.queue._maxsize

    def empty(self):
        return self.queue.empty()

    def full(self):
        return self.queue.full()

    def replenish(self,buffer,b_size):
        with self.lock:
            while not self.queue.full(): 
                batch = buffer.sample(b_size)
                self.queue.put(batch)

    def get(self):
        with self.lock:
            batch = self.queue.get()
            return batch

    def size(self):
        return self.queue.qsize()

