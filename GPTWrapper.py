from threading import Event, Thread
import backoff
import json
import os
import datetime
from time import sleep
from .larknotice import LarkBot
from watchdog.observers import Observer
from typing import List, Callable, Iterable, Dict, Union
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from multiprocessing import Process, Manager, Queue
import inspect
import queue
import openai

class CustomHandler(FileSystemEventHandler):
    def __init__(self, event):
        self.event_to_set = event

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent) and event.src_path.endswith('config.json'):
            with open(event.src_path, 'r') as f:
                try:
                    config = json.load(f)
                except Exception as e:
                    print(f'Error: {e}\ncontinue waiting...')
                    return
            self.event_to_set.set()


class GPTWrapper:
    def __init__(self, config_path, base_wait_time=30, lark_hook=None, bias=0) -> None:
        self.config_path = config_path
        self.bias = bias
        config = json.load(open(self.config_path, 'r', encoding='utf-8'))
        self.key_index = config['key_index']
        self.key_list = config['key_list']
        # add a bias to key_list to support multi thread processing
        self.key_list = [self.key_list[(i - self.bias) % len(self.key_list)] for i in range(len(self.key_list))]
        try:
            self.lark_bot = LarkBot(lark_hook)
        except Exception as e:
            print(f'Error: {e}\nLark notice is not available.')
            print(f'Will run without Lark notice.')
            self.lark_bot = None
        self.base_wait_time = base_wait_time
        self.client = openai.OpenAI(
            api_key=self.key_list[self.key_index].get('api_key', None),
            organization=self.key_list[self.key_index].get('organization', None),
            base_url=self.key_list[self.key_index].get('base_url', None)
            )
        
    def __send_message_periodically(self, stop_event):
        wait_turn = 0
        while not stop_event.is_set():
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.lark_bot.send(f'OpenAI: You exceeded ALL your current quota. Please update `config.json` file to resume.\nTimeStamp: {current_time}')
            sleep(2**wait_turn*self.base_wait_time)
            wait_turn += 1
        
    def set_api_key(self):
        self.key_index += 1
        while self.key_index >= len(self.key_list):
            config = json.load(open(self.config_path, 'r', encoding='utf-8'))
            key_list = config['key_list']
            key_index = config['key_index']
            if self.key_list != key_list:
                self.key_list = key_list
                self.key_list = [self.key_list[(i - self.bias) % len(self.key_list)] for i in range(len(self.key_list))]
                self.key_index = key_index
            else:
                event = Event()
                event_handler = CustomHandler(event)
                observer = Observer()
                observer.schedule(event_handler, os.path.dirname(self.config_path), recursive=False)
                observer.start()
                
                if self.lark_bot:
                    message_thread = Thread(target=self.__send_message_periodically, args=[event])
                    message_thread.start()
                print("Monitoring config.json for changes, main thread is blocked.")
                event.wait()

                print("Config file has changed, main thread continues.")
                observer.stop()
                observer.join()
                if self.lark_bot:
                    message_thread.join()
        self.client = openai.OpenAI(
            api_key=self.key_list[self.key_index].get('api_key', None),
            organization=self.key_list[self.key_index].get('organization', None),
            base_url=self.key_list[self.key_index].get('base_url', None)
            )
        
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def completions_with_backoff(
            self,
            messages,
            engine="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            **kwargs
        ):
        """create a completion with gpt. Currently support `davinci`, `turbo` and `gpt-4`

        Args:
            messages (list): messages sent to `turbo`, `gpt4` or a list of prompts sent to `davinci`. (When using davinci, it is recommended to request in batches. @ref: https://platform.openai.com/docs/guides/rate-limits/error-mitigation)
            engine (str, optional): gpt model. Defaults to "gpt-3.5-turbo".
            temperature (float, optional): Defaults to 0.7.
            max_tokens (int, optional): Defaults to 2048.
            top_p (int, optional): Defaults to 1.
            frequency_penalty (int, optional): Defaults to 0.
            presence_penalty (int, optional): Defaults to 0.

        Raises:
            NotImplementedError: _description_

        Returns:
            response(str) for `turbo` and `gpt-4`
            responses(List[str]) for `davinci`
        """
        openai.api_key = self.key_list[self.key_index]['key']
        if 'org' in self.key_list[self.key_index]:
            openai.organization = self.key_list[self.key_index]['org']
        sleep_Time = 1
        while True:
            try:
                if 'davinci' in engine or 'turbo-instruct' in engine:
                        completion = self.client.completions.create(
                            model=engine,
                            prompt=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            **kwargs
                        )
                        responses = [""]*len(messages)
                        for choice in completion['choices']:
                            responses[choice['index']] = choice['text']
                        return responses
                elif 'turbo' in engine or 'gpt-4' in engine:
                        completion = self.client.chat.completions.create(
                            model=engine,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty
                            **kwargs
                        )
                        return completion["choices"][0]["message"]["content"]
                else:
                    raise NotImplementedError('Currently only support `davinci`, `turbo` and `gpt-4`')
            except openai.RateLimitError as ex:
                if 'Rate limit reached' in str(ex):
                    raise ex
                elif 'exceeded' in str(ex):
                    print(str(ex)+f'\nCurrent api key: {openai.api_key}')
                    self.set_api_key()
                else:
                    print(f'RateLimiteError unhandled...')
                    raise ex
            except openai.BadRequestError as ex:
                if 'have access to' in str(ex):
                    print(ex)
                    self.set_api_key()
                else:
                    raise ex
            except openai.AuthenticationError as ex:
                if 'deactivated' in str(ex):
                    print(f'Api key: {self.key_list[self.key_index]["key"]} has been deactivated. Origin error message: {ex}')
                    self.set_api_key()
                else:
                    raise ex
            except Exception as ex:
                print(ex)
                    # print("##"*5 + ex + "##" *5)
                sleep(sleep_Time)
                sleep_Time *= 2
                if sleep_Time > 1024:
                    print("Sleep time > 1024s")
                    exit(0)

    @staticmethod
    def multi_process_pipeline(config_path: str, processes_num: int, data: Iterable, func: Callable, *args, **kwargs):
        """Execute the function `func` using data and *args, **kwargs

        Args:
            config_path (str): Config file path.
            processes_num (int): The number of processes.
            data (Iterable): The data to process.
            func (Callable): Data processing function.
            args (Any): Additional arguments passed to the function.
            kwargs (Any): Additional keyword arguments passed to the function.

        Returns:
            Any: Return the execution result.
        """
        if os.path.exists(config_path) is False:
            raise FileExistsError(f'Failed to find {config_path}. Please check your file path and try again.')
        elif config_path.endswith('json') is False:
            raise ValueError('Please construct the config file in `JSON` format.')
        
        manager = Manager()
        error_queue = manager.Queue()
        result_queue = [Queue() for _ in range(processes_num)]
        
        lark_hook = kwargs.pop('lark_hook', None)
        
        def __wrapper_func(pid, result_queue: Queue, wrapper, data_chunk, *args, **kwargs):
            try:
                result = func(pid, wrapper, data_chunk, *args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                error_queue.put((pid, e))
        
        chunk_size = round(len(data)/processes_num)
        processes = []
        results = []
        for i in range(processes_num):
            wrapper = GPTWrapper(config_path=config_path, bias=i, lark_hook=lark_hook)
            data_chunk = data[i*chunk_size:(i+1)*chunk_size]
            process = Process(target=__wrapper_func, args=(i, result_queue[i], wrapper, data_chunk, *args), kwargs=kwargs)
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
            
        if not error_queue.empty():
            pid, error = error_queue.get()
            raise type(error)(f"An error occurred in child process {pid}: {str(error)}") from error

        for q in result_queue:
            while not q.empty():
                if type(q.get()) == list:
                    results.extend(q.get())
                else:
                    results.append(q.get())
        
        return results
        
    @staticmethod
    def single_round_multi_process(config_path: str, engine: str, processes_num: int, system_prompts: List, prompts: List, fout: str, **kwargs):
        """Use system prompts and prompts to generate response with multiple processes. If engine is not a chat model, prompt will be formatted
        as `[System Prompt]: {system_prompt}\\n[Prompt]: {prompt}` if system_prompts is not None.

        Args:
            config_path (str): Config file path.
            engine (str): GPT engine.
            processes_num (int): Number of processes to use.
            system_prompts (List): System prompts.
            prompts (List): Prompts.
            fout (str): Output file. `jsonl` recommended.

        Returns:
            List[JSON]: List of results. 
        """
        assert len(system_prompts) == len(prompts)
        def __generate_response(wrapper: GPTWrapper, engine: str, system_prompts: List[str], prompts: List[str], fout: str, **kwargs):
            results = []
            for system_prompt, prompt in zip(system_prompts, prompts):
                if 'davinci' or 'turbo-instruct' in engine:
                    messages = f'[System Prompt]: {system_prompt}\n[Prompt]: {prompt}'
                elif 'gpt-3.5' or 'gpt-4' in engine:
                    messages = [
                        {
                            'role': 'system',
                            'content': system_prompt
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ]
                else:
                    raise NotImplementedError('Currently only support `davinci`, `gpt-3.5` and `gpt-4`')
                response = wrapper.completions_with_backoff(
                    messages=messages,
                    engine=engine,
                    **kwargs
                )
                result = {
                    'system_prompt': system_prompt,
                    'prompt': prompt,
                    'response': response
                }
                results.append(result)
                with open(fout, 'a', encoding='utf-8') as fp:
                    fp.write(json.dumps(result, ensure_ascii=False)+'\n')
                
            return results
        
        chunk_size = round(len(prompts)/processes_num)
        processes = []
        lark_hook = kwargs.pop('lark_hook', None)
        for i in range(processes_num):
            wrapper = GPTWrapper(config_path=config_path, bias=i, lark_hook=lark_hook)
            system_prompts_subset = system_prompts[i*chunk_size:(i+1)*chunk_size]
            prompts_subset = system_prompts[i*chunk_size:(i+1)*chunk_size]
            
            process = Process(target=__generate_response, args=(wrapper, engine, system_prompts_subset, prompts_subset, f'worker{i}_{fout}'), kwargs=kwargs)
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
        
        results = []
        idx = 0
        fp = open(fout, 'a', encoding='utf-8')
        for i in range(processes_num):
            with open(f'worker{i}_{fout}', 'r', encoding='utf-8') as f:
                data = [json.loads(s) for s in f.readlines()]
                for item in data:
                    item['id'] = idx
                    result = json.dumps(item, ensure_ascii=False)
                    results.append(result)
                    fp.write(result + '\n')
                    idx += 1
        fp.close()
        return results
    
    @staticmethod
    def multi_thread_pipeline(config_path: str, threads_num: int, data: Iterable, func: Callable, *args, **kwargs):
        """Execute the function `func` using data and *args, **kwargs

        Args:
            config_path (str): Config file path.
            threads_num (int): The number of threads.
            data (Iterable): The data to process.
            func (Callable): Data processing function.
            args (Any): Additional arguments passed to the function.
            kwargs (Any): Additional keyword arguments passed to the function.

        Returns:
            Any: Return the execution result.
        """
        if os.path.exists(config_path) is False:
            raise FileExistsError(f'Failed to find {config_path}. Please check your file path and try again.')
        elif config_path.endswith('json') is False:
            raise ValueError('Please construct the config file in `JSON` format.')
        
        error_queue = queue.Queue()
        result_queue = queue.Queue()
        
        def __wrapper_func(tid: int, wrapper: GPTWrapper, data_chunk: List, *args, **kwargs):
            try:
                result = func(tid, wrapper, data_chunk, *args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                error_queue.put((tid, e))
        
        chunk_size = round(len(data)/threads_num)
        threads = []
        results = []
        lark_hook = kwargs.pop('lark_hook', None)
        for i in range(threads_num):
            wrapper = GPTWrapper(config_path=config_path, bias=i, lark_hook=lark_hook)
            data_chunk = data[i*chunk_size:(i+1)*chunk_size]
            thread = Thread(target=__wrapper_func, args=(i, wrapper, data_chunk, *args), kwargs=kwargs)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
            
        if not error_queue.empty():
            tid, error = error_queue.get()
            raise type(error)(f"An error occurred in child thread {tid}: {str(error)}") from error

        while not result_queue.empty():
            if type(result_queue.get()) == list:
                results.extend(result_queue.get())
            else:
                results.append(result_queue.get())
        
        return results
    
    @staticmethod
    def single_round_multi_thread(config_path: str, engine: str, threads_num: int, system_prompts: List, prompts: List, fout: str, **kwargs):
        """Use system prompts and prompts to generate response with multiple threads. If engine is not a chat model, prompt will be formatted
        as `[System Prompt]: {system_prompt}\\n[Prompt]: {prompt}` if system_prompts is not None.

        Args:
            config_path (str): Config file path.
            engine (str): GPT engine.
            threads_num (int): Number of threads to use.
            system_prompts (List): System prompts.
            prompts (List): Prompts.
            fout (str): Output file. `jsonl` recommended.

        Returns:
            List[JSON]: List of results. 
        """
        assert len(system_prompts) == len(prompts)
        def __generate_response(wrapper: GPTWrapper, engine: str, system_prompts: List[str], prompts: List[str], fout: str, **kwargs):
            results = []
            if os.path.exists(fout):
                results = [json.loads(s) for s in open(fout, 'r', encoding='utf-8').readlines()]
                system_prompts = system_prompts[len(results):]
            prompts = prompts[len(results):]
            for system_prompt, prompt in zip(system_prompts, prompts):
                if 'davinci' or 'turbo-instruct' in engine:
                    if system_prompt:
                        messages = f'[System Prompt]: {system_prompt}\n[Prompt]: {prompt}'
                    else:
                        messages = prompt
                elif 'gpt-3.5' or 'gpt-4' in engine:
                    if system_prompt:
                        messages = [
                            {
                                'role': 'system',
                                'content': system_prompt
                            },
                            {
                                'role': 'user',
                                'content': prompt
                            }
                        ]
                    else:
                        messages = [
                            {
                                'role': 'user',
                                'content': prompt
                            }
                        ]
                else:
                    raise NotImplementedError('Currently only support `davinci`, `gpt-3.5` and `gpt-4`')
                response = wrapper.completions_with_backoff(
                    messages=messages,
                    engine=engine,
                    **kwargs
                )
                result = {
                    'system_prompt': system_prompt,
                    'prompt': prompt,
                    'response': response
                }
                results.append(result)
                fp.write(json.dumps(result, ensure_ascii=False)+'\n')
                
            return results
        
        chunk_size = round(len(prompts)/threads_num)
        processes = []
        lark_hook = kwargs.pop('lark_hook', None)
        for i in range(threads_num):
            wrapper = GPTWrapper(config_path=config_path, bias=i, lark_hook=lark_hook)
            system_prompts_subset = system_prompts[i*chunk_size:(i+1)*chunk_size]
            prompts_subset = system_prompts[i*chunk_size:(i+1)*chunk_size]
            
            process = Thread(target=__generate_response, args=(wrapper, engine, system_prompts_subset, prompts_subset, f'worker{i}_{fout}'), kwargs=kwargs)
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
        
        results = []
        idx = 0
        fp = open(fout, 'a', encoding='utf-8')
        for i in range(threads_num):
            with open(f'worker{i}_{fout}', 'r', encoding='utf-8') as f:
                data = [json.loads(s) for s in f.readlines()]
                for item in data:
                    item['id'] = idx
                    result = json.dumps(item, ensure_ascii=False)
                    results.append(result)
                    fp.write(result + '\n')
                    idx += 1
        fp.close()
        return results