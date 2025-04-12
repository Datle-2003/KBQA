# import os
# from glob import glob

# import fire
# from loguru import logger

# from common.common_utils import multi_process, read_json
# from common.constant import TOOL_DESC_FULL_FB, TOOL_DESC_FULL_KQAPRO, TOOL_DESC_FULL_METAQA
# from llm_infer_directly import load_test_data
# from tool.action_execution import chat_with_LLM

# """
# the difference between dataset is 
# - how to load test data. we need to sample data by qtype and decide how many data to load.
# - how to load demos. maybe load one for each qtype.
# """


# def load_fewshot_demo_dialog(dataset, qtype=None, entity=None, _4_shot=False):
#     """
#     base: fewshot_demo/{dataset}/dialog/*.txt
#     entity: fewshot_demo/{dataset}/dialog-{entity}-entity/...
#     qtype: fewshot_demo/{dataset}/dialog/{qtype}-[01/02].txt

#     for cwq/kqapro:
#         - `_4_shot` is used to load fixed 4-shot demo to represent all qtype.
#             - kqa: QueryName / QueryRelation / QueryRelationQualifier / Verify
#         - `qtype` is used to load demo by qtype.
#     """
#     dir_patt = f"fewshot_demo/{dataset}/dialog/"
#     if _4_shot:
#         dir_patt = f"fewshot_demo/{dataset}/dialog-4-shot/*.txt"
#     else:
#         if entity:
#             dir_patt = dir_patt[:-1] + f"-{entity}-entity/"
#         if qtype:
#             dir_patt += f"{qtype}-[0-9][0-9].txt"
#         else:
#             dir_patt += "*.txt"

#     logger.warning(f"dir_patt: {dir_patt}")
#     paths = glob(dir_patt)

#     demos = []
#     for p in paths:
#         lines = open(p).readlines()
#         lines = [i for i in lines if not i.startswith("#")]
#         content = "".join(lines).strip()
#         _demos = content.split("\n\n")
#         demos.extend(_demos)

#     if qtype:
#         assert len(demos) == 2, f"if qtype is not None, len(demos) should be 2, but got {len(demos)}"

#     logger.warning(f"len(demos): {len(demos)}")
#     return demos


# def run(dataset, model_name, debug=False, case_num=10, qtype=None, entity=None, fix_4_shot=False):
#     assert entity in [None, "golden"], "entity must be one of None, golden"
#     if entity is not None:
#         logger.warning(f"Using entity: {entity}")

#     if dataset in ["webqsp", "cwq"]:
#         db = "fb"
#         _desc = TOOL_DESC_FULL_FB
#     elif dataset == "kqapro":
#         db = "kqapro"
#         _desc = TOOL_DESC_FULL_KQAPRO
#         assert qtype is not None, "qtype must be provided."
#     elif dataset == "metaqa":
#         db = "metaqa"
#         _desc = TOOL_DESC_FULL_METAQA
#     else:
#         raise ValueError(f"dataset: {dataset} not supported.")

#     skip_ids = []
#     demos = load_fewshot_demo_dialog(dataset=dataset, qtype=qtype, entity=entity, _4_shot=fix_4_shot)

#     tooldesc_demos = _desc + "\n\n" + "\n\n".join(demos)
#     print("tooldesc_demos")
#     print(tooldesc_demos)
#     print()

#     # NOTE: only use test set
#     data = load_test_data(dataset, case_num=case_num)

#     # filter the typed data by qtype prediction
#     if qtype is not None:
#         assert (
#             f"data_preprocess/{dataset}-classification-prediction.json"
#         ), "You need to run question type prediction code first."
#         predicated_qtype = read_json(f"data_preprocess/{dataset}-classification-prediction.json")
#         id_to_pred_qtype = {i["id"]: i["pred_label"] for i in predicated_qtype}
#         data = [i for i in data if id_to_pred_qtype[i["id"]] == qtype]
#         for d in data:
#             d["pred_label"] = qtype

#     print(f"len(data): {len(data)}")

#     # print(data)
#     # logger.info(f"tooldesc ----------------------------------------------------- {tooldesc_demos}")

#     # save to: save-qa-infer-dialog/{dataset}/{setting}/{id}.json
#     _name = dataset + f"-addqtype" if qtype is not None else dataset
#     _name = _name + f"-{entity}" if entity is not None else _name
#     save_dir = f"save-qa-infer-dialog/{_name}/" + model_name.replace("/", "-")
#     logger.info(f"saving to: {save_dir}")

#     if os.path.exists(save_dir):
#         paths = glob(save_dir + "/*.json")
#         skip_ids += [read_json(p)["id"] for p in paths]

#     skip_ids = set(skip_ids)
#     logger.info(f"Skip id: {len(skip_ids)}")
#     data = [d for d in data if d["id"] not in skip_ids]
#     logger.info(f"remain len(data): {len(data)}")

    

#     multi_process(
#         items=data,
#         process_function=chat_with_LLM,
#         cpu_num=1,  # parallel number. NOTE: it consumes money so fast.
#         debug=debug,
#         dummy=True,
#         # func params
#         db=db,
#         model_name=model_name,
#         tooldesc_demos=tooldesc_demos,
#         max_round_num=10,
#         save_dir=save_dir,
#         entity=entity,
#     )


# if __name__ == "__main__":
#     """
#     # case_num: Only run the first case_num cases.
#     export model_name=gpt-4-1106-preview

#     # ---------- webqsp ----------
#     # you can add ` --entity golden` param to run with golden entity.
#     python llm_interactive_kbqa.py --dataset webqsp --model_name ${model_name} --case_num 150

#     # ---------- cwq ----------
#     # you can add ` --entity golden` param to run with golden entity.
#     python llm_interactive_kbqa.py --dataset cwq --model_name ${model_name} --case_num 150 --qtype conjunction
#     python llm_interactive_kbqa.py --dataset cwq --model_name ${model_name} --case_num 150 --qtype composition
#     python llm_interactive_kbqa.py --dataset cwq --model_name ${model_name} --case_num 150 --qtype comparative
#     python llm_interactive_kbqa.py --dataset cwq --model_name ${model_name} --case_num 150 --qtype superlative

#     # ---------- kqapro ----------
#     python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype Count
#     python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype QueryAttr
#     python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype QueryAttrQualifier
#     python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype QueryName
#     python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype QueryRelation
#     python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype QueryRelationQualifier
#     python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype SelectAmong
#     python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype SelectBetween
#     python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype Verify

#     # ---------- metaqa ----------
#     python llm_interactive_kbqa.py --dataset metaqa --model_name ${model_name} --case_num 300


#     # ---------- inference directly with OpenLLM ----------

#     # please refer to the above examples to run the following commands.
#     # `model_name` has to be defined in `from common.constant import LLM_FINETUNING_SERVER_MAP`
#     export model_name=LLMs/mistralai/Mistral-7B-Instruct-v0.2

#     python llm_interactive_kbqa.py --dataset webqsp --model_name ${model_name} --case_num 10
#     python llm_interactive_kbqa.py --dataset cwq --model_name ${model_name} --case_num 10 --qtype conjunction
#     python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype Count --case_num 150
#     python llm_interactive_kbqa.py --dataset metaqa --model_name ${model_name} --case_num 999
#     """
#     fire.Fire(run)



import os
from glob import glob

import fire
from loguru import logger

"""
the difference between dataset is 
- how to load test data. we need to sample data by qtype and decide how many data to load.
- how to load demos. maybe load one for each qtype.
"""

from common.common_utils import multi_process, read_json
from common.constant import (
    TOOL_DESC_FULL_FB,
    TOOL_DESC_FULL_KQAPRO,
    TOOL_DESC_FULL_METAQA,
)
from llm_infer_directly import load_test_data
from tool.action_execution import chat_with_LLM


class KBQARunner:
    def __init__(self, dataset, model_name, case_num=10, qtype=None,entity=None, fix_4_shot=False, debug=False):
        assert entity in [None, "golden"], "Entity must be None or 'golden'"
        self.dataset = dataset
        self.model_name = model_name
        self.case_num = case_num
        self.qtype = qtype
        self.entity = entity
        self.fix_4_shot = fix_4_shot
        self.debug = debug

        self.db, self.tool_desc = self._get_tool_desc()
        self.save_dir = self._get_save_dir()
        self.skip_ids = set()

    def _get_tool_desc(self):
        if self.dataset in ["webqsp", "cwq"]:
            return "fb", TOOL_DESC_FULL_FB
        elif self.dataset == "kqapro":
            assert self.qtype, "qtype must be provided for kqapro"
            return "kqapro", TOOL_DESC_FULL_KQAPRO
        elif self.dataset == "metaqa":
            return "metaqa", TOOL_DESC_FULL_METAQA
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
    def _get_save_dir(self):
        name = self.dataset
        if self.qtype:
            name += "-addqtype"
        if self.entity:
            name += f"-{self.entity}"
        return f"save-qa-infer-dialog/{name}/{self.model_name.replace('/', '-')}"

    def _load_demo_dialogs(self):
        if self.fix_4_shot:
            pattern = f"fewshot_demo/{self.dataset}/dialog-4-shot/*.txt"
        else:
            base = f"fewshot_demo/{self.dataset}/dialog"
            if self.entity:
                base += f"-{self.entity}-entity"
            pattern = base + "/"
            pattern += f"{self.qtype}-[0-9][0-9].txt" if self.qtype else "*.txt"

        logger.warning(f"Loading demos from: {pattern}")
        demos = []
        for path in glob(pattern):
            with open(path, "r") as f:
                lines = [line for line in f if not line.startswith("#")]
                content = "".join(lines).strip()
                demos.extend(content.split("\n\n"))

        if self.qtype:
            assert len(demos) == 2, f"Expected 2 demos for qtype '{self.qtype}', got {len(demos)}"
        logger.warning(f"Loaded {len(demos)} demos")
        return demos

    def _prepare_data(self):
        data = load_test_data(self.dataset, case_num=self.case_num)

        if self.qtype:
            pred_file = f"data_preprocess/{self.dataset}-classification-prediction.json"
            assert os.path.exists(pred_file), f"Missing prediction file: {pred_file}"
            preds = read_json(pred_file)
            id_to_pred = {p["id"]: p["pred_label"] for p in preds}
            data = [d for d in data if id_to_pred.get(d["id"]) == self.qtype]
            for d in data:
                d["pred_label"] = self.qtype

        return data

    def _skip_existing(self, data):
        if os.path.exists(self.save_dir):
            paths = glob(f"{self.save_dir}/*.json")
            self.skip_ids = {read_json(p)["id"] for p in paths}

        logger.info(f"Skipping {len(self.skip_ids)} previously processed items")
        return [d for d in data if d["id"] not in self.skip_ids]

    def run(self):
        logger.info(f"Running KBQA with model: {self.model_name}")
        os.makedirs(self.save_dir, exist_ok=True)

        demos = self._load_demo_dialogs()
        tool_demos = self.tool_desc + "\n\n" + "\n\n".join(demos)

        print("=== TOOL + DEMOS ===\n", tool_demos, "\n")

        data = self._prepare_data()
        data = self._skip_existing(data)
        logger.info(f"Final data size: {len(data)}")

        multi_process(
            items=data,
            process_function=chat_with_LLM,
            cpu_num=1,
            debug=self.debug,
            dummy=True,
            db=self.db,
            model_name=self.model_name,
            tooldesc_demos=tool_demos,
            max_round_num=10,
            save_dir=self.save_dir,
            entity=self.entity,
        )


def main(**kwargs):
    runner = KBQARunner(**kwargs)
    runner.run()


if __name__ == "__main__":
    
    """
    # case_num: Only run the first case_num cases.
    export model_name=gpt-4-1106-preview

    # ---------- webqsp ----------
    # you can add ` --entity golden` param to run with golden entity.
    python llm_interactive_kbqa.py --dataset webqsp --model_name ${model_name} --case_num 150

    # ---------- cwq ----------
    # you can add ` --entity golden` param to run with golden entity.
    python llm_interactive_kbqa.py --dataset cwq --model_name ${model_name} --case_num 150 --qtype conjunction
    python llm_interactive_kbqa.py --dataset cwq --model_name ${model_name} --case_num 150 --qtype composition
    python llm_interactive_kbqa.py --dataset cwq --model_name ${model_name} --case_num 150 --qtype comparative
    python llm_interactive_kbqa.py --dataset cwq --model_name ${model_name} --case_num 150 --qtype superlative

    # ---------- kqapro ----------
    python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype Count
    python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype QueryAttr
    python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype QueryAttrQualifier
    python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype QueryName
    python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype QueryRelation
    python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype QueryRelationQualifier
    python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype SelectAmong
    python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype SelectBetween
    python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype Verify

    # ---------- metaqa ----------
    python llm_interactive_kbqa.py --dataset metaqa --model_name ${model_name} --case_num 300


    # ---------- inference directly with OpenLLM ----------

    # please refer to the above examples to run the following commands.
    # `model_name` has to be defined in `from common.constant import LLM_FINETUNING_SERVER_MAP`
    export model_name=LLMs/mistralai/Mistral-7B-Instruct-v0.2

    python llm_interactive_kbqa.py --dataset webqsp --model_name ${model_name} --case_num 10
    python llm_interactive_kbqa.py --dataset cwq --model_name ${model_name} --case_num 10 --qtype conjunction
    python llm_interactive_kbqa.py --dataset kqapro --model_name ${model_name} --qtype Count --case_num 150
    python llm_interactive_kbqa.py --dataset metaqa --model_name ${model_name} --case_num 999
    """

    fire.Fire(main)
        