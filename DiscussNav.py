import numpy as np
import cv2
import json
import math
import MatterSim
import torch
import time
import networkx as nx
from PIL import Image
from lavis.models import load_model_and_preprocess
from tasks.R2R.utils import load_nav_graphs
from tqdm import tqdm
import requests

import os
import sys
import logging

log_file = sys.argv[0].replace("py","log")
if os.path.exists(log_file): os.remove(log_file)
logging.basicConfig(
    format='%(asctime)s - %(filename)s/%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
    filemode="a"
)
logger = logging.getLogger("navLLM")
logger.addHandler(logging.FileHandler(filename=log_file))

# Simulator image parameters
WIDTH = 1024
HEIGHT = 1024
VFOV = 60
HFOV = VFOV*WIDTH/HEIGHT
TEXT_COLOR = [230, 40, 40]
ERROR_MARGIN = 3

# Set up the simulator
sim = MatterSim.Simulator()
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(math.radians(VFOV))
sim.setDepthEnabled(False)
sim.setDiscretizedViewingAngles(True)
sim.initialize()

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

from lavis.models import load_model_and_preprocess
from ram.models import ram
from ram import inference_ram
from ram import get_transform
        

def gpt_response(prompt, model, num_output, temperature=1):
    header = {
        "Content-Type":"application/json",
        "Authorization": "Your OpenAI key"
    }
    post_dict = {
        "model": model,
        "messages": prompt,
        "temperature": temperature,
        "n":num_output,
    }
    while True:
        try:
            r = requests.post("URL", json=post_dict, headers=header)
            return r.json()['choices']
        except Exception as e:
            print(e)
            continue

def get_nearest(shortest_distances, goal_id, path):
    near_id = path[0]
    near_d = shortest_distances[near_id][goal_id]
    for item in path:
        d = shortest_distances[item][goal_id]
        if d < near_d:
            near_id = item
            near_d = d
            
    return near_id

def get_all_distances(dataset):
    all_scans = [item["scan"] for item in dataset]
    all_graphs = load_nav_graphs(all_scans)
    all_distances = {}
    for scan, G in all_graphs.items():  # compute all shortest paths
        all_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    return all_distances

def open_image(image_path):
    state = sim.getState()[0]
    rgb = np.array(state.rgb, copy=False)
    cv2.imwrite(image_path, rgb)
    raw_img = Image.open(image_path).convert("RGB")

    return raw_img

class Instruction_Analysis_Experts:
    def detect_actions(self, instruction):
        instruction = gpt_response([{"role": "user", "content": f"Please translate the sentence \"{instruction}\" into English. Your translation is: "}], "gpt-4", 1)[0]["message"]["content"]
        prompt = [
            {"role": "system", "content": "You are an action decomposition expert. Your task is to detect all actions in the given navigation instruction. You need to ensure the integrity of each action."},
            {"role": "user", "content": f"Can you decompose actions in the instruction \"{instruction}\"? Actions: "},
        ]

        return gpt_response(prompt, "gpt-4", 1)[0]["message"]["content"]

    def detect_landmarks(self, actions):
        actions = actions.replace("\n", " ")
        prompt = [
            {"role": "system", "content": "You are a landmark extraction expert. Your task is to detect all landmarks in the given navigation instruction. You need to ensure the integrity of each landmarks."},
            {"role": "user", "content": f"Can you extract landmarks in the instruction \"{actions}\"? Landmarks: "},
        ]

        return gpt_response(prompt, "gpt-4", 1)[0]["message"]["content"]

class Vision_Perception_Experts:
    def __init__(self, ram_path, current_rgb_path="current_view.jpg", view_record_path="cache_files/view_cache.json"):
        self.current_rgb_path = current_rgb_path
        self.view_record_path = view_record_path
        if os.path.exists(self.view_record_path): 
            with open(self.view_record_path, "r", encoding="utf-8") as file:
                self.view_record = json.load(file)
        else:
            self.view_record = {}
        
        while True:
            try:
                self.instructblip_model, self.instructblip_vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=device)
                self.ram_transform = get_transform(image_size=384)
                self.ram_model = ram(pretrained=ram_path, image_size=384, vit='swin_l').eval().to(device)
                break
            except:
                continue

    def ram_img_tagging(self, image):
        ram_img = self.ram_transform(image).unsqueeze(0).to(device)
        img_tags = inference_ram(ram_img, self.ram_model)[0]

        return img_tags

    def instructblip_description(self, image, scene_questions, img_tags):
        instruct_blip_img = self.instructblip_vis_processors["eval"](image).unsqueeze(0).to(device)
        prompt_list = ["Describe this indoor scene in details"]

        response_list = []
        for start_idx in range(0, 999, 3):
            end_idx = len(prompt_list) if start_idx + 3 > len(prompt_list) else start_idx + 3
            prompts = prompt_list[start_idx:end_idx]
            if prompts == []:
                break
            batch_images = torch.cat(tuple([instruct_blip_img]*len(prompts)))
            response_list.extend(self.instructblip_model.generate({"image": batch_images, "prompt": prompts}))

        return " ".join(response_list)

    def observe_view(self, direction_idx, instruction, landmarks, navigable_vps, navigable_vps_elevation, scene_questions, history_vp):
        raw_img = open_image(self.current_rgb_path)
        current_vp = sim.getState()[0].location.viewpointId
        navigable_vps_str = ", ".join(navigable_vps)
        view_observe_idx = f"{current_vp} -> {navigable_vps_str} (Elevation 0)"
        
        if view_observe_idx not in self.view_record.keys():
            img_tags = self.ram_img_tagging(raw_img)
            instructblip_img_info = self.instructblip_description(raw_img, scene_questions, img_tags)
            view_observation = f"Scene Description: {instructblip_img_info} Scene Objects: {img_tags}; "
            self.view_record[view_observe_idx] = view_observation
        else:
            view_observation = self.view_record[view_observe_idx]
            
        relative_elevation = navigable_vps_elevation[0]
        if relative_elevation < -0.1:
            elevation_flag = "(lower position indicates down stairs)"
        elif relative_elevation > 0.1:
            elevation_flag = "(higher position indicates up stairs)"
        else:
            elevation_flag = ""
            
        direction_idx, navigable_vps_temp = str(direction_idx), navigable_vps[0]
        if len(set(navigable_vps).intersection(set(history_vp))) > 0:
            observe_result = f"Direction {direction_idx} {elevation_flag} Navigable Viewpoint ID: {navigable_vps_temp} (Passed Area) Elevation: Eye Level "  + view_observation
        else:
            observe_result = f"Direction {direction_idx} {elevation_flag} Navigable Viewpoint ID: {navigable_vps_temp} Elevation: Eye Level " + view_observation
        
        if "stair" in instruction or "the steps" in instruction:
            sim.makeAction([0], [0], [-math.radians(30)])
            raw_img = open_image(self.current_rgb_path)
            view_observe_idx = f"{current_vp} -> {navigable_vps_str} (Elevation -30)"
            if view_observe_idx not in self.view_record.keys():
                img_tags = self.ram_img_tagging(raw_img)
                view_observation = f"Scene Objects: {img_tags}; "
                self.view_record[view_observe_idx] = view_observation
            else:
                view_observation = self.view_record[view_observe_idx]
            observe_result += f"Elevation: Look Down " + view_observation
            sim.makeAction([0], [0], [math.radians(30)])

        logger.info(observe_result)

        return observe_result

    def observe_environment(self, start_vp, direction_list, instruction, landmarks, scene_questions, nav_history):
        observe_results = []
        view_vp_list = []
        history_vp = [start_vp]+[item["viewpoint"] for item in nav_history]
        logger.info("history VP: "+str(history_vp))
        
        for direction_idx, direction in enumerate(direction_list): 
            state = sim.getState()[0]
            navigable_vps = [loc.viewpointId for loc in state.navigableLocations[1:]]
            navigable_vps_elevation = [loc.rel_elevation for loc in state.navigableLocations[1:]]
            if navigable_vps == [] or (navigable_vps in view_vp_list) or (set(navigable_vps).intersection(set(history_vp)) == set(navigable_vps)):
            # if navigable_vps == [] or (navigable_vps in view_vp_list):
                sim.makeAction([0], [direction*math.radians(360/len(direction_list))], [0])
                observe_results.append("")
                continue

            view_vp_list.append(navigable_vps)

            observe_result = self.observe_view(direction_idx, instruction, landmarks, navigable_vps, navigable_vps_elevation, scene_questions, history_vp)
            observe_results.append(observe_result) 

            sim.makeAction([0], [direction*math.radians(360/len(direction_list))], [0])

        with open(self.view_record_path, "w", encoding="utf-8") as file:
            json.dump(self.view_record, file, indent=2)

        return observe_results

class Completion_Estimation_Experts:
    def summarize_observation(self, curr_observe, landmarks):
        direction_id = int(curr_observe.split("Navigable Viewpoint")[0].replace("Direction","").replace("(lower position indicates down stairs)","").replace("(higher position indicates up stairs)","").strip())
        directions = ["Front, range(right 0 to right 30)", "Font Right, range(right 30 to right 60)", "Right, range(right 60 to right 90)", "Right, range(right 90 to right 120)", "Rear Right, range(right 120 to right 150)", "Rear Right, range(right 150 to right 180)",
                    "Rear Left, range(left 180 to left 150)", "Rear Left, range(left 150 to left 120)", "Left, range(left 120 to left 90)", "Left, range(left 90 to left 60)", "Front Left, range(left 60 to left 30)", "Front Left, range(left 30 to left 0)"]
        direction = directions[direction_id]
        curr_observe = "Scene Description"+curr_observe.split("Scene Description")[1]
        prompt = [
            {"role": "system", "content": "You are a trajectory summary expert. Your task is to simplify environment description as short and clear as possible."},
            {"role": "user", "content": f"Given Environment Description \"{curr_observe}\", Summarization:"}
        ]
        
        return f"Direction {direction} " + gpt_response(prompt, "gpt-4", 1)[0]["message"]["content"]

    def summarize_thought(self, thought):
        prompt = [
            {"role": "system", "content": "You are a trajectory summary expert. Your task is to simplify navigation thought process as short and clear as possible."},
            {"role": "user", "content": f"Given Thought Process \"{thought}\", Summarization:"}
        ]
        
        return gpt_response(prompt, "gpt-4", 1)[0]["message"]["content"]

    def save_history(self, next_vp, thought, curr_observe, nav_history, landmarks): 
        curr_observe = self.summarize_observation(curr_observe, landmarks)
        thought = self.summarize_thought(thought)
        nav_history.append({
            "viewpoint": next_vp,
            "observation": curr_observe,
            "thought": thought
        })
        
        return nav_history
    
    def review_history(self, nav_history):
        nav_history_str = " -> ".join(["Step "+str(idx+1)+" Observation: "+item["observation"]+" Thought: "+item["thought"] for idx, item in enumerate(nav_history)])
        logger.info("History: "+nav_history_str)
        
        return nav_history_str

    def estimate_completion(self, actions, landmarks, history_traj):
        prompt = [
            {"role": "system", "content": "You are a completion estimation expert. Your task is to estimate what actions in the instruction have been executed based on navigation history and landmarks. \
                All actions in the instruction are given following the temporal order. Your answer includes two parts: \"Thought\" and \"Executed Actions\". \
                In the \"Thought\", you must follow procedures to analyze as detailed as possible what actions have been executed: \
                (1) What given landmarks of actions have appeared in the navigation history? \
                (2) Analyze the direction change at each step in the navigation history. \
                (3) Estimate each action in the instruction based on each step in the navigation history to check their completion. \
                In the \"Executed Actions\", you must only write down actions that have been executed without other words. \
                You must strictly refer original actions in the given instruction to estimate."},
            {"role": "user", "content": f"Given Navigation History \"{history_traj}\" and Landmarks \"{landmarks}\", estimate what actions in instruction \"{actions}\" have been executed."}   
        ]
        response = gpt_response(prompt, "gpt-4", 1)[0]["message"]["content"]
        if "Executed Actions" in response:
            logger.info("Executed Actions "+response)
            return response.split("Executed Actions")[1].strip()
        else:
            return response

class DiscussNav_Agent:
    def __init__(self, num_predictions=5, num_retry=5):
        self.num_predictions = num_predictions
        self.num_retry = num_retry
    
    def pred_vp(self, current_step, instruction, actions, landmarks, history_traj, estimation, observation):
        pred_definition = "You are a navigation agent who follows instruction to move in an indoor environment with the least action steps. \
            I will give you one instruction and tell you landmarks. I will also give you navigation history and estimation of executed actions for reference. \
            You can observe current environment by scene descriptions, scene objects and possible existing landmarks in different directions around you. \
            Each direction contains navigable viewpoints you can move to. Your task is to predict moving to which navigable viewpoint. \
            Note that environment direction that contains more landmarks mentioned in the instruction is usually the better choice for you. \
            If you are required to go up stairs, you need to move to direction with higher position. If you are required to go down stairs, you need to move to direction with lower position. \
            You are encouraged to move to new viewpoints to explore environment while avoid revisiting accessed viewpoints in non-essential situations. \
            If you feel struggling to find the landmark or execute the action, you can try to execute the subsequent action and find the subsequent landmark. \
            Your answer includes two parts: \"Thought\" and \"Prediction\". In the \"Thought\", you should think as detailed as possible following procedures: \
            (1) Check whether the latest executed action has been completed by comparing current environment and landmark in the latest executed action. \
            (2) Determine the action you should execute and landmark you should reach now. If the latest executed action have not been completed, \
            you should continue to execute it. Otherwise, you should execute the next action in the given instruction. \
            (3) Analyze which direction in the current environment is most suitable to execute the action you decide and explain your reason. \
            (4) Predict moving to which navigable viewpoint based on your thought process. \
            Then, please make decision on the next viewpoint in the \"Prediction\". You must only answer next viewpoint ID in the \"Prediction\" without other words. \
            Your decision is very important, must make it very carefully."
        input_info = f"Step {current_step} Instruction: {instruction} ({actions}) Landmarks: {landmarks} Navigation History: {history_traj} \
            Estimation of Executed Actions: {estimation} Current Environment: {observation} -> Thought: ... Prediction: ..."
        decision_prompt = [
            {"role": "system", "content": pred_definition},
            {"role": "user", "content": input_info}
        ]
    
        break_flag = True
        for _ in range(self.num_retry):
            effective_prediction, thought_list = [], []
            batch_responses = gpt_response(decision_prompt, "gpt-4", self.num_predictions)
            for item_response in batch_responses:
                decision_reasoning = item_response["message"]["content"]
                if "Prediction:" not in decision_reasoning:
                    continue
                logger.info(decision_reasoning)
                pred_thought = decision_reasoning.split("Prediction:")[0].strip()
                pred_vp = decision_reasoning.split("Prediction:")[1].strip().replace("\"","").replace("'","").replace("\n","").replace(".","").replace("*","")

                if len(pred_vp) != 32:
                    logger.info(f"{pred_vp} Length Problem")
                    continue
                if pred_vp not in str(observation):
                    logger.info(f"{pred_vp} not in the candidates")
                    continue

                effective_prediction.append(pred_vp)
                thought_list.append(pred_thought)
            
            if len(effective_prediction) == self.num_predictions:
                break_flag = False
                break

        return effective_prediction, thought_list, break_flag

    def make_action(self, direction_list, observation, next_vp):
        for direction_idx, direction in enumerate(direction_list):
            state = sim.getState()[0]
            navigable_vps = [loc.viewpointId for loc in state.navigableLocations[1:]]
            if navigable_vps == []:
                sim.makeAction([0], [direction*math.radians(360/len(direction_list))], [0])
                continue
            if next_vp == navigable_vps[0]:
                sim.makeAction([1], [0], [0])
                logger.info("Move to "+next_vp)
                return observation[direction_idx]
            
            sim.makeAction([0], [direction*math.radians(360/len(direction_list))], [0])

class Decision_Testing_Experts:
    def thought_fusion(self, predictions, thoughts):
        matched_dict = dict()
        for pred, thought in zip(predictions, thoughts):
            if pred not in matched_dict.keys():
                matched_dict[pred] = []
            matched_dict[pred].append(thought)
            
        for key, value in matched_dict.items():
            multiple_thoughts = "; ".join(["Thought "+str(idx+1)+": "+thought for idx, thought in enumerate(value)])
            prompt = [
                {"role": "system", "content": "You are a thought fusion expert. Your task is to fuse given thought processes \
                    into one thought. You need to reserve key information related to actions, landmarks, direction changes. You should only answer fused thought without other words."},
                {"role": "user", "content": f"Can you help me fuse the thoughts leading to the same movement direction? The thoughts are :{multiple_thoughts}, Fused thought: "}   
            ]
            one_thought = gpt_response(prompt, "gpt-4", 1)[0]["message"]["content"]  
            logger.info(f"Pred: {key} Fused Thought: {one_thought}")
            matched_dict[key] = one_thought
            
        return matched_dict      
    
    def test_decisions(self, fused_pred_thought, observation, instruction):
        if len(fused_pred_thought.keys()) == 1:
            for key, value in fused_pred_thought.items():
                return key, value
        else:
            fused_pred_thought_ = "; ".join(["Navigation Viewpoint ID: "+key+" Thought: "+value for key, value in fused_pred_thought.items()])
            decision_prompt = [
                {"role": "system", "content": "You are a decision testing expert. Your task is to evaluate the feasibility of each movement \
                    prediction based on thought process and environment. Then, you will make a final decision about navigation viewpoint ID without other words."},
                {"role": "user", "content": f"Can you help me make a final decision? The Observation: {observation}, Navigation Instruction: {instruction}, {fused_pred_thought_}, Final Decision: "}   
            ]
            for _ in range(3):
                next_vp = gpt_response(decision_prompt, "gpt-4", 1, temperature=0)[0]["message"]["content"].strip()
                if len(next_vp) != 32:
                    logger.info(f"{next_vp} Length Problem")
                    continue
                if next_vp not in fused_pred_thought.keys():
                    logger.info(f"{next_vp} not in the candidates")
                    continue
                break
        
        return next_vp, fused_pred_thought[next_vp]

def eval_pred(final_vp, nav_history, shortest_distances, eval_cache, instruction, scanId, des_vp, gt_path):
    path = [item["viewpoint"] for item in nav_history]+[final_vp]
                
    nav_error = shortest_distances[scanId][final_vp][des_vp]
    nearest_position = get_nearest(shortest_distances[scanId], gt_path[-1], path)
    oracle_error = shortest_distances[scanId][nearest_position][gt_path[-1]]
    
    trajectory_length = np.sum([shortest_distances[scanId][a][b] for a, b in zip(path[:-1], path[1:])])
    gt_lengths = np.sum([shortest_distances[scanId][a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])

    success = float(nav_error < ERROR_MARGIN)
    oracle_success = float(oracle_error < ERROR_MARGIN)
    spl = success * gt_lengths / max(trajectory_length, gt_lengths, 0.01)

    eval_cache[instruction] = {
        "trajectory_length": trajectory_length,
        "nav_error": nav_error,
        "success": success,
        "oracle_success": oracle_success,
        "spl": spl,
        "time": "First",
    }
    
    tl = np.mean([item["trajectory_length"] for item in eval_cache.values()])
    ne = np.mean([item["nav_error"] for item in eval_cache.values()])
    sr = np.mean([item["success"] for item in eval_cache.values()]) * 100
    osr = np.mean([item["oracle_success"] for item in eval_cache.values()]) * 100
    spl_rate = np.mean([item["spl"] for item in eval_cache.values()]) * 100
    
    logger.info("                           ")
    logger.info(str(eval_cache[instruction]))
    logger.info("********************************")
    logger.info("Total Length (TL) : "+str(tl))
    logger.info("Navigation Error (NE) : "+str(ne))
    logger.info("Oracle Success Rate (OSR) : "+str(osr))
    logger.info("Success Rate (SR) : " + str(sr))
    logger.info("SPL: "+str(spl_rate))

    return eval_cache


def main(dataset_name, ram_path):
    with open(f"tasks/{dataset_name}/data/{dataset_name}_val_unseen.json", "r") as file:
        dataset = json.load(file)

    shortest_distances = get_all_distances(dataset)

    if not os.path.exists(f"cache_files/{dataset_name}"):
        os.makedirs(f"cache_files/{dataset_name}")

    actions_cache_path = f"./cache_files/{dataset_name}/actions_cache.json"
    if os.path.exists(actions_cache_path): 
        with open(actions_cache_path, "r", encoding="utf-8") as file:
            actions_cache = json.load(file)
    else:
        actions_cache = {}
    
    eval_cache_path = f"./cache_files/{dataset_name}/eval_cache.json"
    if os.path.exists(eval_cache_path): 
        with open(eval_cache_path, "r", encoding="utf-8") as file:
            eval_cache = json.load(file)
    else:
        eval_cache = {}
    
    # Create multiple experts for visual language navigation
    ia_experts = Instruction_Analysis_Experts()
    vp_experts = Vision_Perception_Experts(ram_path)
    ce_experts = Completion_Estimation_Experts()
    dt_experts = Decision_Testing_Experts()
    
    # Create DiscussNav agent
    discussnav_agent = DiscussNav_Agent()
        
    for item in tqdm(dataset):
        scanId, gt_path, start_heading, instructions = item["scan"], item["path"], item["heading"], item["instructions"]
        start_vp, des_vp = gt_path[0], gt_path[-1]

        for instruction in instructions:
            sim.newEpisode([scanId], [start_vp], [start_heading], [0])
            logger.info("======================================================================================")
            logger.info("Instruction: "+instruction)
            logger.info("GT Path: "+str(gt_path))
            logger.info("Path Length: "+str(len(gt_path)))
            
            scene_questions = []
            nav_history = []

            actions, landmarks = "", ""
            if instruction not in actions_cache.keys():
                actions = ia_experts.detect_actions(instruction)
                landmarks = ia_experts.detect_landmarks(actions)
                actions_cache[instruction] = {"actions": actions, "landmarks": landmarks}
                with open(actions_cache_path, "w", encoding="utf-8") as f2:
                    json.dump(actions_cache, f2, indent=2)
            else:
                actions = actions_cache[instruction]["actions"]
                landmarks = actions_cache[instruction]["landmarks"]
            logger.info("Actions: "+actions)
            
            step_length = 5 if len(actions.split("\n")) <= 5 else 7 # Hyper-parameter
            for current_step in range(step_length):
                logger.info(f"------------------------------Step {current_step}------------------------------")
                observation = vp_experts.observe_environment(start_vp, [1]*12, instruction, landmarks, scene_questions, nav_history)
                logger.info("Landmarks: "+landmarks)
                
                history_traj = ce_experts.review_history(nav_history) if len(nav_history) > 0 else "Step 0 start position."
                estimation = ce_experts.estimate_completion(actions, landmarks, history_traj)
                
                predictions, thoughts, break_flag = discussnav_agent.pred_vp(current_step, instruction, actions, landmarks, history_traj, estimation, observation)
                if break_flag: 
                    break
                
                fused_pred_thought = dt_experts.thought_fusion(predictions, thoughts)
                next_vp, thought = dt_experts.test_decisions(fused_pred_thought, observation, instruction)
                
                curr_observe = discussnav_agent.make_action([1]*12, observation, next_vp)
                nav_history = ce_experts.save_history(next_vp, thought, curr_observe, nav_history, landmarks)
                    
            final_vp = sim.getState()[0].location.viewpointId
            eval_cache = eval_pred(final_vp, nav_history, shortest_distances, eval_cache, instruction, scanId, des_vp, gt_path)
            with open(eval_cache_path, "w", encoding="utf-8") as file:
                json.dump(eval_cache, file, indent=2)

        
if __name__ == "__main__":
    dataset_name = "R2R"
    ram_path = "recognize-anything/ckpt/ram_swin_large_14m.pth"
    main(dataset_name, ram_path)