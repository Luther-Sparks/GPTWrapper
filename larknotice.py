from datetime import datetime
import os
import requests
import functools
import socket
import traceback

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def lark_sender(webhook_url: str=None, content: str=None):
    """Lark sender wrapper: execute func, send a Lark notification with the end status
    (sucessfully finished or crashed) at the end. Also send a Lark notification before
    executing func.

    Args:
        webhook_url (str, optional): The webhook URL to access your lark robot. Defaults to None.
        content (str, optional): The message you want to send. Defaults to None .
    """
    bot = LarkBot(webhook_url)

    def decorator_sender(func):
        @functools.wraps(func)
        def wrapper_sender(*args, **kwargs):
            print(f'start...')

            start_time = datetime.now()
            host_name = socket.gethostname()
            func_name = func.__name__
            
            # simply send content after finishing running function
            if content is not None:
                try:
                    value = func(*args, **kwargs)
                    bot.send(content=content)
                except Exception as ex:
                    end_time = datetime.now()
                    elapsed_time = end_time - start_time
                    contents = ["Your training has crashed ☠️",
                                'Machine name: %s' % host_name,
                                'Main call: %s' % func_name,
                                'Starting date: %s' % start_time.strftime(DATE_FORMAT),
                                'Crash date: %s' % end_time.strftime(DATE_FORMAT),
                                'Crashed training duration: %s\n\n' % str(elapsed_time),
                                "Here's the error:",
                                '%s\n\n' % ex,
                                "Traceback:",
                                '%s' % traceback.format_exc()]
                    bot.send(content='\n'.join(contents))
                    raise ex
                return
                
            

            # Handling distributed training edge case.
            # In PyTorch, the launch of `torch.distributed.launch` sets up a RANK environment variable for each process.
            # This can be used to detect the master process.
            # See https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py#L211
            # Except for errors, only the master process will send notifications.
            if 'RANK' in os.environ:
                master_process = (int(os.environ['RANK']) == 0)
                host_name += ' - RANK: %s' % os.environ['RANK']
            else:
                master_process = True

            if master_process:
                contents = ['Your program has started 🎬',
                            'Machine name: %s' % host_name,
                            'Main call: %s' % func_name,
                            'Starting date: %s' % start_time.strftime(DATE_FORMAT)]
                bot.send(content='\n'.join(contents))

            try:
                value = func(*args, **kwargs)

                if master_process:
                    end_time = datetime.now()
                    elapsed_time = end_time - start_time
                    contents = ["Your program is complete 🎉",
                                'Machine name: %s' % host_name,
                                'Main call: %s' % func_name,
                                'Starting date: %s' % start_time.strftime(DATE_FORMAT),
                                'End date: %s' % end_time.strftime(DATE_FORMAT),
                                'Training duration: %s' % str(elapsed_time)]

                    try:
                        str_value = str(value)
                        contents.append('Main call returned value: %s'% str_value)
                    except:
                        contents.append('Main call returned value: %s'% "ERROR - Couldn't str the returned value.")

                    bot.send(content='\n'.join(contents))

                return value

            except Exception as ex:
                end_time = datetime.now()
                elapsed_time = end_time - start_time
                contents = ["Your training has crashed ☠️",
                            'Machine name: %s' % host_name,
                            'Main call: %s' % func_name,
                            'Starting date: %s' % start_time.strftime(DATE_FORMAT),
                            'Crash date: %s' % end_time.strftime(DATE_FORMAT),
                            'Crashed training duration: %s\n\n' % str(elapsed_time),
                            "Here's the error:",
                            '%s\n\n' % ex,
                            "Traceback:",
                            '%s' % traceback.format_exc()]
                bot.send(content='\n'.join(contents))
                raise ex

        return wrapper_sender

    return decorator_sender


class LarkBot:
    def __init__(self, hook_url=None) -> None:
        if hook_url is None:
            if 'LARK_HOOK' in os.environ:
            # 提取LARK_HOOK的值
                hook_url = os.environ['LARK_HOOK']
            else:
                raise ValueError('Failed to get Lark hook url. Add url to environment or pass it as an argument to class `LarkBot`.')
        self.hook_url = hook_url

    def send(self, content: str) -> None:
        timestamp = int(datetime.now().timestamp())

        params = {
            "timestamp": timestamp,
            # "sign": sign,
            "msg_type": "text",
            "content": {"text": content},
        }
        resp = requests.post(url=self.hook_url, json=params)
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") and result["code"] != 0:
            print(result["msg"])


@lark_sender()
def train_your_model():
    import time
    time.sleep(10)
    return {'loss': 0.9}

if __name__ == '__main__':
    train_your_model()