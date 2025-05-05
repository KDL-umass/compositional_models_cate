# read a sample plan and calculate the individual times for each activity.
import json as json
import re
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open('{}/queries/jsons/operations_schema_all_features.json'.format(ROOT_DIR)) as f:
    operations_schema = json.load(f)
def parseNumberedSubPlanName(name):
    m = re.search(r'^([^ ]+) (\d+) \(returns ([^)]+)\)$', name)
    
    if(m is None):
        return None
    name, ID, dollarIDs =  m.groups()
    # index substring from 1 to remove the $ sign.



    returns = [int(x[1]) for x in dollarIDs.split(",")]
    return {"type": name, "ID": int(ID), "returns": returns}
        
def parseFilter(filter):
    if filter is None:
        return []
    pattern = re.compile(r'\$\d+')
    m = re.findall(pattern, filter)
    
    if m is None:
        return []
    
    return [int(x[1]) for x in m]

def sumTotalTime(nodes):
                sum = 0
                for n in nodes:
                    sum += n.data["Total Time"]
                return sum

def sumSelfBlocks(nodes, key):
    sum = 0
    for n in nodes:
        sum += n.data[key]
    return sum

            

class Node():
    def __init__(self, name, parent = None, level = 0, key = 1):
        self.name = name
        self.parent = parent
        self.children = []
        self.left_child = None
        self.right_child = None
        self.data = {}
        self.filter_refs = []
        self.filter_nodes = []
        self.cte_nodes = []
        self.cte_scans = []
        self.level = level
        self.key = key
        self.features = {}
        

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
    def add_child(self, child):
        self.children.append(child)

class PlanTree():
    def __init__(self, name = None, query_id = 0, memory_level = 0, plan_dict = None):
        if name is not None:
            self.root = Node(name)
        else:
            self.root = self.create_tree_from_dict(plan_dict)
        self.query_id = query_id
        self.memory_level = memory_level
        self.training_data = {}
        self.json_data = {"query_id": self.query_id, "memory_level": memory_level, "Plans": [{}]}
        self.operations = [name]

    def __str__(self):
        return self.root.name
    
    def __repr__(self):
        return self.root.name
    
    def add_child_helper(self, parent, child, child_type = "left"):
        if child_type == "left":
            parent.left_child = child
            parent.children.append(child)
            child.parent = parent
            child.level = parent.level + 1
            child.key = parent.key * 2
        else:
            parent.right_child = child
            parent.children.append(child)
            child.parent = parent
            child.level = parent.level + 1
            child.key = parent.key * 2 + 1

        # print("=======Updated Tree=======") 
        # self.print_tree(self.root)
        # print("==========================")
        # add left child to the last node in the tree.
    
    def add_child(self, parent_node, child_name, child_type = "left"):
        # print("=======Current Tree=======") 
        # self.print_tree(self.root)
        # print("==========================")
        # add left child to the last node in the tree.
        # print("Adding {} child {} to parent {}".format(child_type, child_name, parent_node.name))
        child_node = Node(child_name)
        curr_node = self.root
        self.operations.append(child_name)
        # print(parent_node.name, parent_node.level, parent_node.key, self.root.name, self.root.level, self.root.key)
        
        if parent_node.level == 0 and parent_node.name != self.root.name:
            print("parent not found at root level")
            return
        
        if parent_node.level == 0 and parent_node.name == self.root.name and parent_node.key == self.root.key:
            self.add_child_helper(curr_node, child_node, child_type)
            return child_node
        
        level = 0
        found = False
        while level < parent_node.level - 1:
            curr_node = curr_node.left_child
            level += 1

    
        # print(curr_node.name, curr_node.level, curr_node.key)

        if curr_node.left_child != None:
            # print("checking left child")
            if curr_node.left_child.name == parent_node.name and curr_node.left_child.key == parent_node.key:
                found = True
                # print("parent {} found in left node with key {}".format(parent_node.name, parent_node.key))
                self.add_child_helper(curr_node.left_child, child_node, child_type)
        
        if curr_node.right_child != None and found == False:
            # print("checking right child")
            if curr_node.right_child.name == parent_node.name and curr_node.right_child.key == parent_node.key:
                found = True
                # print("parent {} found in right node with key {}".format(parent_node.name, parent_node.key))
                self.add_child_helper(curr_node.right_child, child_node, child_type)
            else:
                print("parent not found in right node")
                return None
        
        if found == False:
            print("parent not found in left and right node")
            return None
        
        
        return child_node

        
    def print_tree(self, node, print_keys = [], level = 0):
        if len(print_keys) == 0:
            print_string = "{} {} {} {} ".format("  "*level, node.name, node.level, node.key, )
        else:
            print_string = "{} {} {} {} ".format("  "*level, node.name, node.level, node.key, )
            for key in print_keys:
                data = node.features.get(key, None)
                if data is not None:
                    # if data is float then print with 3 decimal places
                    if isinstance(data, float):
                        print_string += "{} {:.3f} ".format(key, node.features[key])
                    else:
                        print_string += "{} {} ".format(key, node.features[key])
        print(print_string)
        # if len(node.children) == 0 or level == 0:
        #     print_string = "{} {} {} {} ".format("  "*level, node.name, node.level, node.key)
        #     print(print_string)
        # for child in node.children:
        #     print_string = "{} {} {} {} ".format("  "*(level+1), node.name, node.level, node.key)
        for child in node.children:
            self.print_tree(child, print_keys = print_keys, level = level + 1)

    def convert_tree_to_dict(self, node):
        if node is None:
            return
        
        if node.name not in self.training_data:
            self.training_data[node.name] = []

        features_dict = {"query_id": self.query_id, "memory_level": self.memory_level, "parent_node": node.parent.name if node.parent is not None else "None"}
        for feature in node.features:
            features_dict[feature] = node.features[feature]
        self.training_data[node.name].append(features_dict)

        self.convert_tree_to_dict(node.left_child)
        self.convert_tree_to_dict(node.right_child)

    def convert_tree_to_json(self, node, current_json_data):
        if node is None:
            return
        current_json_data["name"] = node.name
        current_json_data["Plans"] = []

        for feature in node.features:
            current_json_data[feature] = node.features[feature]
        
        for i, child in enumerate(node.children):
            current_json_data["Plans"].append({})
            self.convert_tree_to_json(child, current_json_data["Plans"][i])

    def create_tree_from_dict(self, plan_node, parent = None):
        # create a tree from the dictionary.
        # create a root node.
        root_node = Node(plan_node['name'], parent = parent)
        root_node.name = plan_node['name']
        root_node.data = plan_node
        if "Plans" in plan_node:
            for count, plan in enumerate(plan_node['Plans']):
                root_node.add_child(self.create_tree_from_dict(plan, root_node))
                if count == 0:
                    root_node.left_child = root_node.children[0]
                elif count == 1:
                    root_node.right_child = root_node.children[1]
                else:
                    # print("Three children found for a node.")
                    self.three_children = True

        return root_node

    def compare_plans(self, node1, node2):
        if node1.name != node2.name:
            return False
        
        if len(node1.children) != len(node2.children):
            return False
        
        for i in range(len(node1.children)):
            if not self.compare_plans(node1.children[i], node2.children[i]):
                return False
        
        return True
  
               

    # def create_tree_from_dict(self, plan_json):


    

            
class PlanExplainer():
    # initialize plan explainer with a plan node which is a dictionary.
    def __init__(self, query_id, treatment_id, run_id, plan_dict, set_total_time = True):
        self.plan_dict = plan_dict
        self.three_children = False
        # create plan node as a copy of the original plan node.
        self.plan_tree = self.create_tree_from_dict(self.plan_dict)
        self.all_ops_set = set()
        self.depth = self.calcDepth(self.plan_tree)
        self.save_all_op_names(self.plan_tree)
        self.all_dfs = {}
        self.query_id = query_id
        self.treatment_id = treatment_id
        self.run_id = run_id
        if set_total_time:
            self.setTotalTime(self.plan_tree)
        self.all_operations = []
        self.specific_operation = {}
        self.training_data = {}
        self.all_predicates = []
        self.training_features = {}
        

    def create_tree_from_dict(self, plan_node, parent = None):
        # create a tree from the dictionary.
        # create a root node.
        root_node = Node(plan_node['Node Type'], parent = parent)
        root_node.data = plan_node
        if "Plans" in plan_node:
            for count, plan in enumerate(plan_node['Plans']):
                root_node.add_child(self.create_tree_from_dict(plan, root_node))
                if count == 0:
                    root_node.left_child = root_node.children[0]
                elif count == 1:
                    root_node.right_child = root_node.children[1]
                else:
                    # print("Three children found for a node.")
                    self.three_children = True

        return root_node
    
    def save_all_op_names(self, node):
        self.all_ops_set.add(node.name)
        for child in node.children:
            self.save_all_op_names(child)
    
    def calcDepth(self, node):
        if len(node.children) == 0:
            return 1
        else:
            return 1 + max([self.calcDepth(child) for child in node.children])

    
    def setTotalTime(self, node):
        for child in node.children:
            self.setTotalTime(child)
        
        node.data["Total Time"] = node.data["Actual Total Time"]
        node.data["Self Time"] = 0
        node.data["Self Shared Read Blocks"] = 0
        node.data["Self Shared Hit Blocks"] = 0
        node.data["Self Shared Dirtied Blocks"] = 0
        node.data["Self Shared Written Blocks"] = 0


    def calcActualLoops(self, node, gather_node = None):
        if (node.name == "Gather" or node.name == "Gather Merge") and ("Actual Loops" in node.data):
            gather_node = node

        for child in node.children:
            self.calcActualLoops(child, gather_node)

        if "Actual Loops" not in node.data:
            return 
        
        loops = node.data["Actual Loops"]
        if gather_node is not None:
            loops = gather_node.data["Actual Loops"]
        
        node.data["Total Time"] = node.data["Total Time"] * loops
        
            
    

    def setFilterRefs(self, node, root = None):
        
        if root is None:
            root = node
        for child in node.children:
            self.setFilterRefs(child, root)

        if "Subplan Name" not in node.data:
            return
        
        sp = parseNumberedSubPlanName(node.data["Subplan Name"])

        if sp is None:
            return
        
        def visit(fn2):
            for child in fn2.children:
                visit(child)
            filters = []
            if "Filter" in fn2.data:
                filters.append(fn2.data["Filter"])
            
            if "One-Time Filter" in fn2.data:
                filters.append(fn2.data["One-Time Filter"])

            for filter in filters:
                parsedFilterList = parseFilter(filter)
                
                for ids in sp["returns"]:
                    if ids in parsedFilterList:
                        node.filter_refs.append(fn2)
                        fn2.filter_nodes.append(node)
        visit(root)
        return
    
    def setCTERefs(self, node):
        for child in node.children:
            self.setCTERefs(child)
        
        if node.name != "CTE Scan":
            return
        
        parent = node 
        while parent is not None:
            cte_node = None
            for child in parent.children:
                if child.data["Parent Relationship"] == "InitPlan" and child.data["Subplan Name"] == "CTE " + node.data["CTE Name"]:
                    cte_node = child

            if cte_node is not None:
                node.cte_nodes.append(cte_node)
                cte_node.cte_scans.append(node)
                
            parent = parent.parent

    def calcFilter(self, node, key):
        for child in node.children:
            self.calcFilter(child, key)

        init_value = node.data[key]

        if len(node.filter_refs) == 0:
            return
        
        childCount = len(node.filter_refs)
        for child in node.filter_refs:
            delta = init_value/childCount
            p = child
            while p is not None:
                for c in p.children:
                    if c == node:
                        return
                val = p.data[key]
                p.data[key] = val - delta
                p = p.parent

        
    
    def calcCTE(self, node, key):
        for i in reversed(range(len(node.children))):
            self.calcCTE(node.children[i], key)

        init_value = node.data[key]
        if len(node.cte_scans) == 0:
            return
        
        sum_scans = 0
        for cte_scan in node.cte_scans:
            sum_scans += cte_scan.data[key]

        for cte_scan in node.cte_scans:
            for n in cte_scan.cte_nodes:
                if n.parent == cte_scan:
                    return 
            
            new_val = cte_scan.data[key] * (1 - init_value/sum_scans)
            delta = cte_scan.data[key] - new_val

            # print("CTE Scan: " + cte_scan.name + " " + str(cte_scan.data[key]) + " " + str(new_val) + " " + str(delta))
            cte_scan.data[key] = new_val

            p = cte_scan.parent
            break_loop = False
            while p is not None:
                # print("Parent fixing", p.name)
                for c in p.children:
                    if c == node:
                        # print(c.name, node.name)
                        # print("======= Hello Returning ======")
                        break_loop = True
                        break

                if break_loop:
                    break
                p_val = p.data[key]
                # print("Parent fixing Name: ", p.name, p_val, delta)
                p.data[key] = p_val - delta
                p = p.parent

    def calcParallelAppendTime(self, node, gather = None, scale = 1):
        if node.name == "Gather":
            gather = node
            
        if scale != 1:
            node.data["Total Time"] = node.data["Total Time"] * scale
            
        if node.name == "Append" and node.data["Parallel Aware"] == True and gather is not None:
            scale = gather.data["Total Time"]/ sumTotalTime(node.children)

        for child in node.children:
            self.calcParallelAppendTime(child, gather, scale)


    def calcChildBoost(self, node):
        for child in node.children:
            self.calcChildBoost(child)
        
        childTotalTime = sumTotalTime(node.children)
        if childTotalTime > node.data["Total Time"]:
            node.data["Total Time"] = childTotalTime


    def setSelfTime(self, node):
        for child in node.children:
            self.setSelfTime(child)
        
        node.data["Self Time"] = node.data["Total Time"] - sumTotalTime(node.children)

    def setSharedBlocks(self, node):
        for child in node.children:
            self.setSharedBlocks(child)
        
        node.data["Self Shared Read Blocks"] = node.data["Shared Read Blocks"] - sumSelfBlocks(node.children, "Shared Read Blocks")
        node.data["Self Shared Hit Blocks"] = node.data["Shared Hit Blocks"] - sumSelfBlocks(node.children, "Shared Hit Blocks")
        node.data["Self Shared Dirtied Blocks"] = node.data["Shared Dirtied Blocks"] - sumSelfBlocks(node.children, "Shared Dirtied Blocks")
        node.data["Self Shared Written Blocks"] = node.data["Shared Written Blocks"] - sumSelfBlocks(node.children, "Shared Written Blocks")

    def adjustSharedBlocks(self, node):
        for child in node.children:
            self.adjustSharedBlocks(child)
        if "Actual Loops" in node.data:
            if node.data["Actual Loops"] != 0:
                node.data["Self Shared Read Blocks"] = node.data["Self Shared Read Blocks"]/node.data["Actual Loops"]
                node.data["Self Shared Hit Blocks"] = node.data["Self Shared Hit Blocks"]/node.data["Actual Loops"]
                node.data["Self Shared Dirtied Blocks"] = node.data["Self Shared Dirtied Blocks"]/node.data["Actual Loops"]
                node.data["Self Shared Written Blocks"] = node.data["Self Shared Written Blocks"]/node.data["Actual Loops"]
            

    # print tree with levels 
    def print_tree(self, node, print_keys = [], level = 0):
        if len(print_keys) == 0:
            print_keys = ["Actual Startup Time", "Actual Total Time"]
        print_string = "{} {} ".format("  "*level, node.name)
        for key in print_keys:
            data = node.data.get(key, None)
            if data is not None:
                # if data is float then print with 3 decimal places
                if isinstance(data, float):
                    print_string += "{} {:.3f} ".format(key, node.data[key])
                else:
                    print_string += "{} {} ".format(key, node.data[key])
        print(print_string)
        for child in node.children:
            self.print_tree(child, print_keys = print_keys, level = level + 1)

    # print the reference nodes
    def print_ref_nodes(self, node):
        
        for refs in node.filter_refs:
            print(node.name, node.data.get("Subplan Name", None), "Referenced By  ", refs.name, refs.data.get("Filter", None), refs.data.get("One-Time Filter", None))
        
        for child in node.children:
            self.print_ref_nodes(child)

    # print the filter nodes
    def print_filter_nodes(self, node):
        for n in node.filter_nodes:
            print(node.name, node.data.get("Filter", None), node.data.get("One-Time Filter", None), "Refers  ", n.name, n.data.get("Subplan Name", None))

        for child in node.children:
            self.print_filter_nodes(child)

    def print_cte_scans(self, node):
        for n in node.cte_scans:
            print(node.name, node.data.get("SubPlan Name", None), node.data.get("Parent Relationship", None), "Contains CTE Scans  ", n.name, n.data.get("CTE Name", None))

        for child in node.children:
            self.print_cte_scans(child)

    def print_cte_nodes(self, node):
        for n in node.cte_nodes:
            print(node.name, node.data.get("Actual Total Time", None), node.data.get("CTE Name", None), "Has Init Node  ", n.name, n.data.get("SubPlan Name", None), n.data.get("Parent Relationship", None))

        for child in node.children:
            self.print_cte_nodes(child)

    def get_total_time(self, node):
        total_time = node.data["Self Time"]
        for child in node.children:
            total_time += self.get_total_time(child)
        return total_time
    
    def check_negative_time(self, node):
        if node.data["Self Time"] < 0:
            return True
        for child in node.children:
            if self.check_negative_time(child):
                return True
        return False
    
    def get_df_per_operation(self, node):
        op_columns = operations_schema.get(node.name)
        if node.name not in self.all_dfs:
            self.all_dfs[node.name] = [[self.query_id, self.treatment_id, self.run_id] + [node.data.get(col, None) for col in op_columns]]
        else:
            self.all_dfs[node.name].append([self.query_id, self.treatment_id, self.run_id] + [node.data.get(col, None) for col in op_columns])
        for child in node.children:
            self.get_df_per_operation(child)

    def convert_tree_to_dict(self, node):
        post_processed_dict = {}
        for key, value in node.data.items():
            if key != "Plans":
                post_processed_dict[key] = value
        post_processed_dict["Plans"] = []
        for child in node.children:
            post_processed_dict["Plans"].append(self.convert_tree_to_dict(child))

        return post_processed_dict
    
    def convert_tree_to_simple_dict(self, node):
        simple_dict = {}
        simple_dict["name"] = node.name
        simple_dict["Plans"] = []
        for child in node.children:
            simple_dict["Plans"].append(self.convert_tree_to_simple_dict(child))
        return simple_dict
    
    

    
    def get_all_operations(self, node):
        self.all_operations.append([node.name, node.data.get("Actual Rows", None), node.data.get("Plan Rows", None), node.data.get("Plan Width", None), node.data.get("Relation Name", None), node.data.get("Index Name", None), node.data.get("Parallel Aware", None)])
        
        for child in node.children:
            self.get_all_operations(child)

    def get_specific_operation(self, node, op):
        if node.name == op:
            if op not in self.specific_operation:
                self.specific_operation[op] = []
            self.specific_operation[op].append([node.data])
        
        for child in node.children:
            self.get_specific_operation(child, op)

    def get_training_features(self, node, ops_schema):
        if node.name not in self.training_features:
            self.training_features[node.name] = []
        dict_json = {}
        for feature in ops_schema[node.name]["pre_execution_features"]:
            dict_json[feature] = node.data.get(feature, None)

        for feature in ops_schema[node.name]["post_execution_features"]:
            dict_json[feature] = node.data.get(feature, None)

        
        dict_json["left_child_input_actual_rows"] = None
        dict_json["left_child_input_rows"] = None
        dict_json["left_child_total_time"] = None
        dict_json["left_child_startup_time"] = None
        dict_json["right_child_input_actual_rows"] = None
        dict_json["right_child_input_rows"] = None
        dict_json["right_child_total_time"] = None
        dict_json["right_child_startup_time"] = None
        
        if node.left_child is not None:
            left_node = node.left_child
            dict_json["left_child_total_time"] = left_node.data["Actual Total Time"]
            dict_json["left_child_startup_time"] = left_node.data["Actual Startup Time"]
            dict_json["left_child_input_rows"] = left_node.data["Plan Rows"]
            if "Actual Loops" in left_node.data:
                dict_json["left_child_input_actual_rows"] = left_node.data["Actual Rows"] * left_node.data["Actual Loops"]
            else:
                dict_json["left_child_input_actual_rows"] = left_node.data["Actual Rows"]
        
    
        if node.right_child is not None:
            right_node = node.right_child
            dict_json["right_child_total_time"] = right_node.data["Actual Total Time"]
            dict_json["right_child_startup_time"] = right_node.data["Actual Startup Time"] 
            dict_json["right_child_input_rows"] = right_node.data["Plan Rows"]
            if "Actual Loops" in right_node.data:
                dict_json["right_child_input_actual_rows"] = right_node.data["Actual Rows"] * right_node.data["Actual Loops"]
            else:
                dict_json["right_child_input_actual_rows"] = right_node.data["Actual Rows"]

        self.training_features[node.name].append(dict_json)
    
        for child in node.children:
            self.get_training_features(child, ops_schema)


    def get_training_data(self, node, ops_schema):
        if node.name not in self.training_data:
            self.training_data[node.name] = []
        dict_json = {}
            
        dict_json["query_id"] = self.query_id
        # print(self.query_id)
        dict_json["treatment_id"] = self.treatment_id
        dict_json["run_id"] = self.run_id
        dict_json["parent_node"] = node.parent.name if node.parent is not None else None
        for feature in ops_schema[node.name]["pre_execution_features"]:
            dict_json[feature] = node.data.get(feature, None)

        for feature in ops_schema[node.name]["post_execution_features"] + ["I/O Read Time", "I/O Write Time", "Self Shared Read Blocks", "Self Shared Hit Blocks", "Self Shared Dirtied Blocks", "Self Shared Written Blocks"] :
            if feature in ["Filter", "Output"]:
                continue
            dict_json[feature] = node.data.get(feature, None)
        dict_json["left_child_input_actual_rows"] = None
        dict_json["left_child_input_rows"] = None
        dict_json["left_child_total_time"] = None
        dict_json["left_child_startup_time"] = None
        dict_json["right_child_input_actual_rows"] = None
        dict_json["right_child_input_rows"] = None
        dict_json["right_child_total_time"] = None
        dict_json["right_child_startup_time"] = None
        
        if node.left_child is not None:
            left_node = node.left_child
            dict_json["left_child_total_time"] = left_node.data["Actual Total Time"]
            dict_json["left_child_startup_time"] = left_node.data["Actual Startup Time"]
            dict_json["left_child_input_rows"] = left_node.data["Plan Rows"]
            if "Actual Loops" in left_node.data:
                dict_json["left_child_input_actual_rows"] = left_node.data["Actual Rows"] * left_node.data["Actual Loops"]
            else:
                dict_json["left_child_input_actual_rows"] = left_node.data["Actual Rows"]
        
    
        if node.right_child is not None:
            right_node = node.right_child
            dict_json["right_child_total_time"] = right_node.data["Actual Total Time"]
            dict_json["right_child_startup_time"] = right_node.data["Actual Startup Time"] 
            dict_json["right_child_input_rows"] = right_node.data["Plan Rows"]
            if "Actual Loops" in right_node.data:
                dict_json["right_child_input_actual_rows"] = right_node.data["Actual Rows"] * right_node.data["Actual Loops"]
            else:
                dict_json["right_child_input_actual_rows"] = right_node.data["Actual Rows"]

        self.training_data[node.name].append(dict_json)
    
        for child in node.children:
            self.get_training_data(child, ops_schema)

    def get_all_predicates(self, node):
        # get all predicates from the query plan
        
        if "Filter" in node.data:
            self.all_predicates.append({"Node Type": node.name, "Filter": node.data["Filter"]})

        if "Hash Cond" in node.data:
            self.all_predicates.append({"Node Type": node.name, "Filter": node.data["Hash Cond"]})

        if "Join Filter" in node.data:
            self.all_predicates.append({"Node Type": node.name, "Filter": node.data["Join Filter"]})

        if "Merge Cond" in node.data:
            self.all_predicates.append({"Node Type": node.name, "Filter": node.data["Merge Cond"]})

        if "Index Cond" in node.data:
            self.all_predicates.append({"Node Type": node.name, "Filter": node.data["Index Cond"]})

        for child in node.children:
            self.get_all_predicates(child)

    def compare_plans(self, node1, node2):
        if node1.name != node2.name:
            return False
        
        if len(node1.children) != len(node2.children):
            return False
        
        for i in range(len(node1.children)):
            if not self.compare_plans(node1.children[i], node2.children[i]):
                return False
        
        return True

    

def get_plan_explainer(query_id, treatment_id, run_id, sample_plan):
    # # create a plan explainer object.
    plan_explainer = PlanExplainer(query_id, treatment_id, run_id, sample_plan)
    plan_explainer.setFilterRefs(plan_explainer.plan_tree)
    plan_explainer.setCTERefs(plan_explainer.plan_tree)
    plan_explainer.calcActualLoops(plan_explainer.plan_tree)
    plan_explainer.calcFilter(plan_explainer.plan_tree, "Total Time")
    plan_explainer.calcCTE(plan_explainer.plan_tree, "Total Time")
    plan_explainer.calcParallelAppendTime(plan_explainer.plan_tree)
    plan_explainer.calcChildBoost(plan_explainer.plan_tree)
    plan_explainer.setSelfTime(plan_explainer.plan_tree)
    plan_explainer.setSharedBlocks(plan_explainer.plan_tree)
    plan_explainer.adjustSharedBlocks(plan_explainer.plan_tree)
    return plan_explainer