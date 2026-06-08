from torch.utils.data import Dataset
from tqdm import tqdm
import json

templates = dict(
    atlas="Please think step by step, and put your final answer within \\boxed{}.",
)

def _ensure_image_placeholder(text: str) -> str:
    if not isinstance(text, str):
        return text
    if "<image>" in text:
        return text
    return "<image>\n" + text

class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def preprocess_data(self, data, input_template=None, input_key="input", apply_chat_template=None, system_prompt="longcot") -> str:
        has_vlm_processor = self.processor is not None
        if has_vlm_processor:
            if system_prompt == 'notrigger':
                trigger = ""
            elif system_prompt == 'elaborate':
                trigger = f"\n\n{templates['elaborate']}"
            elif system_prompt == 'elaborate_rethink':
                trigger = f"\n\n{templates['elaborate_rethink']}"
            elif system_prompt == 'rethink':
                trigger = f"\n\n{templates['rethink']}"
            else:
                trigger = f"\n\n{templates[system_prompt]}"

            q = data['question']
            img = data.get('image', None) or data.get('image_path', None)
            imglist = []
            if img is None or img == "":
                pass
            elif isinstance(img, list):
                if data.get('is_video', False):
                    imglist = [dict(type='video', video=img)]
                else:
                    imglist = [dict(type='image', image=imm) for imm in img if imm]
            else:
                imglist = [dict(type='image', image=img)]

            if len(imglist) > 0:
                q = _ensure_image_placeholder(q)

            chat = [dict(role='user', content=imglist + [dict(type='text', text=q + trigger)])]
            if 'qid' in data:
                chat.append(dict(qid=data['qid']))

            prompt = json.dumps(chat, indent=2)
        elif input_key=='question':
            chat = [{"role": "system", "content": templates["default"]},
                {"role": "user", "content": data['question']}]
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        elif input_key=='messages':
            chat = data[input_key]
            if len(chat)>1: 
                chat[0] = dict(role='system', content=templates[system_prompt]) # replace 
            else: 
                if system_prompt in templates:
                    chat.insert(0, dict(role='system', content=templates[system_prompt]))
                else: print(f'!!!! warning: {system_prompt} not in templates')
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        elif apply_chat_template:
            chat = data[input_key]
            if isinstance(chat, str):
                chat = [{"role": "user", "content": chat}]
            else: # messages 
                if len(chat)>1: 
                    chat[0] = dict(role='system', content=templates[system_prompt]) # replace 
                else: chat.insert(0, dict(role='system', content=templates[system_prompt]))
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            prompt = data[input_key]
            input_template = templates[system_prompt]
            if system_prompt in ['none']:
                chat = [dict(role='system', content=templates[system_prompt]),
                        dict(role='user', content=prompt)
                        ]
                prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            else:
                prompt = input_template.format(prompt)
        return prompt

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        is_eval=False,
        processor=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.processor = processor
        self.is_eval = is_eval
        
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        controlled_shuffle = getattr(self.strategy.args, "controlled_shuffle", 0)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        
        system_prompt = getattr(self.strategy.args, "system_prompt", "none")
        do_vlm = getattr(self.strategy.args, "train_vlm", False)
        if apply_chat_template:
            apply_chat_template = self.processor.apply_chat_template if do_vlm else self.tokenizer.apply_chat_template

        self.prompts = []
        repeat = 1 if controlled_shuffle==0 or is_eval else controlled_shuffle
        for _ in range(repeat):
            for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
                prompt = self.preprocess_data(data, input_template, input_key, apply_chat_template, system_prompt)
                self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]