"""
Pure Python/PyTorch implementation of SVP (Set-Valued Predictor)
translated from the original C++ code: https://github.com/tfmortie/setvaluedprediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import heapq
import math
import copy
import weakref
from typing import List, Optional

# Enumeration of set-valued predictor types
class SVPType(Enum):
    FB = 0
    DG = 1
    SIZECTRL = 2
    ERRORCTRL = 3
    LAC = 4
    RAPS = 5
    CSVPHF = 6
    CRSVPHF = 7

# Parameter container (similar to the C++ 'param' struct)
class SVPParam:
    def __init__(self, svptype=SVPType.FB, beta=1, delta=1.6, gamma=0.6, size=1,
                 error=0.05, rand=False, lambda_=0.0, k=1, c=1):
        self.svptype = svptype
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.size = size
        self.error = error
        self.rand = rand
        self.lambda_ = lambda_
        self.k = k
        self.c = c

# Node in the hierarchical classifier tree.
# Parent pointers are stored in a separate dictionary to avoid registration.
class HNode(nn.Module):
    def __init__(self, par=None):
        super(HNode, self).__init__()
        self.clf = None         # This will hold an nn.Sequential module when set.
        self.y: List[int] = []  # List of labels for this node.
        self.chn: List['HNode'] = []  # List of child nodes.
        # Store non-module metadata in _meta to avoid cycle registration.
        self._meta = {}
        self._meta['par'] = par  # Parent SVP module.
        self._meta['parent'] = None  # Parent node (will be set as weakref).

    def addch(self, in_features, dp, y: List[int], id):
        # If children exist, check whether y is a subset of one of them.
        if self.chn:
            found_child = None
            for child in self.chn:
                if set(y).issubset(set(child.y)):
                    found_child = child
                    break
            if found_child is not None:
                found_child.addch(in_features, dp, y, id)
            else:
                new_node = HNode(par=self._meta['par'])
                new_node.y = y
                new_node._meta['parent'] = weakref.ref(self)
                self.chn.append(new_node)
                tot_len = sum(len(child.y) for child in self.chn)
                if tot_len == len(self.y):
                    # Create a unique label string for registration.
                    ystr = "child_" + " ".join(map(str, y)) + "_" + str(id)
                    clf = nn.Sequential(
                        nn.Dropout(dp),
                        nn.Linear(in_features, len(self.chn))
                    )
                    new_node.clf = clf
                    self._meta['par'].add_module(ystr, clf)
        else:
            new_node = HNode(par=self._meta['par'])
            new_node.y = y
            new_node._meta['parent'] = weakref.ref(self)
            self.chn.append(new_node)
            if len(new_node.y) == len(self.y):
                ystr = "child_" + " ".join(map(str, y)) + "_" + str(id)
                clf = nn.Sequential(
                    nn.Dropout(dp),
                    nn.Linear(in_features, 1)
                )
                new_node.clf = clf
                self._meta['par'].add_module(ystr, clf)

    def forward(self, input, criterion, y_ind=0):
        # Computes loss at this node if multiple children exist.
        loss = torch.tensor(0.0, device=input.device)
        target = torch.tensor([y_ind], dtype=torch.long, device=input.device)
        if self.chn and self.clf is not None:
            out = self.clf(input)
            loss = loss + criterion(out, target)
        return loss

# Wrapper for priority queue nodes.
class QNode:
    def __init__(self, node: HNode, prob: float):
        self.node = node
        self.prob = prob
    def __lt__(self, other: 'QNode'):
        # Reverse order: higher probability has higher priority.
        return self.prob < other.prob

# Main SVP module.
class SVP(nn.Module):
    def __init__(self, in_features, num_classes, dp, hstruct=None):
        """
        If hstruct is empty (or None) the model is flat.
        Otherwise, hstruct (a list of lists) defines the hierarchical structure.
        """
        super(SVP, self).__init__()
        self.num_classes = num_classes
        self.hstruct = None  # Optional hierarchy structure (can be a Tensor)
        self.sr_map = {}     # For dynamic programming (not fully implemented here)
        if hstruct is None or (isinstance(hstruct, list) and len(hstruct) == 0):
            # Flat model: create root with a single predictor mapping to all classes.
            self.root = HNode(par=self)
            clf = nn.Sequential(
                nn.Dropout(dp),
                nn.Linear(in_features, self.num_classes)
            )
            self.root.clf = clf
            # Register the flat classifier under "root_clf" so self.root remains our HNode.
            self.add_module("root_clf", clf)
            self.root.y = []
            self.root.chn = []
        else:
            if isinstance(hstruct, torch.Tensor):
                self.hstruct = hstruct.float()
                # Treat as flat even though hstruct is provided.
                self.root = HNode(par=self)
                clf = nn.Sequential(
                    nn.Dropout(dp),
                    nn.Linear(in_features, self.num_classes)
                )
                self.root.clf = clf
                self.add_module("root_clf", clf)
                self.root.y = []
                self.root.chn = []
            else:
                # Hierarchical model: first element defines root labels.
                self.root = HNode(par=self)
                self.root.y = hstruct[0]
                self.root.chn = []
                for i in range(1, len(hstruct)):
                    self.root.addch(in_features, dp, hstruct[i], i)

    def forward(self, input, target=None):
        """
        Overloaded forward pass.
          - If target is a list of lists (hierarchical targets), traverse the tree.
          - If target is a Tensor, use flat mode.
          - If target is None, return class probabilities.
        """
        if target is not None:
            if isinstance(target, list) and all(isinstance(t, list) for t in target):
                # Hierarchical forward pass.
                loss = torch.tensor(0.0, device=input.device)
                criterion = nn.CrossEntropyLoss()
                for bi in range(input.size(0)):
                    node = self.root
                    for yi in target[bi]:
                        loss = loss + node.forward(input[bi].view(1, -1), criterion, yi)
                        if node.chn and yi < len(node.chn):
                            node = node.chn[yi]
                        else:
                            break
                return loss / input.size(0)
            elif isinstance(target, torch.Tensor):
                # Flat forward pass.
                criterion = nn.CrossEntropyLoss()
                out = self.root.clf(input)
                return criterion(out, target)
        else:
            # Inference: return probabilities.
            if not self.root.y:
                out = self.root.clf(input)
                return F.softmax(out, dim=1)
            else:
                # Hierarchical inference: traverse the tree via a priority queue.
                batch_probs = torch.zeros(input.size(0), self.num_classes, device=input.device)
                for bi in range(input.size(0)):
                    q = []
                    heapq.heappush(q, QNode(self.root, 1.0))
                    while q:
                        current = heapq.heappop(q)
                        if len(current.node.y) == 1:
                            leaf = current.node.y[0]
                            batch_probs[bi, leaf] = current.prob
                        else:
                            out = current.node.clf(input[bi].view(1, -1))
                            probs = F.softmax(out, dim=1).cpu().detach().numpy()[0]
                            for i, child in enumerate(current.node.chn):
                                heapq.heappush(q, QNode(child, current.prob * probs[i]))
                return batch_probs

    def predict(self, input):
        """
        Returns top-1 prediction.
        Flat mode: argmax of the flat classifier.
        Hierarchical mode: traverse the tree by choosing the max–probability child.
        """
        if not self.root.y:
            out = self.root.clf(input)
            return torch.argmax(out, dim=1).cpu().tolist()
        else:
            preds = []
            for bi in range(input.size(0)):
                node = self.root
                while len(node.y) > 1 and node.chn:
                    out = node.clf(input[bi].view(1, -1))
                    ch = torch.argmax(out, dim=1).item()
                    node = node.chn[ch]
                preds.append(node.y[0] if node.y else None)
            return preds

    # -------------------------
    # Set-valued prediction functions (simplified versions)
    def predict_set_fb(self, input, beta, c):
        p = SVPParam(svptype=SVPType.FB, beta=beta, c=c)
        return self.predict_set(input, p)

    def predict_set_dg(self, input, delta, gamma, c):
        p = SVPParam(svptype=SVPType.DG, delta=delta, gamma=gamma, c=c)
        return self.predict_set(input, p)

    def predict_set_size(self, input, size, c):
        p = SVPParam(svptype=SVPType.SIZECTRL, size=size, c=c)
        return self.predict_set(input, p)

    def predict_set_error(self, input, error, c):
        p = SVPParam(svptype=SVPType.ERRORCTRL, error=error, c=c)
        return self.predict_set(input, p)

    def predict_set_lac(self, input, error, c):
        p = SVPParam(svptype=SVPType.LAC, error=error, c=c)
        return self.predict_set(input, p)

    def predict_set_raps(self, input, error, rand, lambda_, k, c):
        p = SVPParam(svptype=SVPType.RAPS, error=error, rand=rand, lambda_=lambda_, k=k, c=c)
        return self.predict_set(input, p)

    def predict_set_csvphf(self, input, error, rand, lambda_, k, c):
        p = SVPParam(svptype=SVPType.CSVPHF, error=error, rand=rand, lambda_=lambda_, k=k, c=c)
        return self.predict_set(input, p)

    def predict_set_crsvphf(self, input, error, rand, lambda_, k):
        p = SVPParam(svptype=SVPType.CRSVPHF, error=error, rand=rand, lambda_=lambda_, k=k)
        return self.predict_set(input, p)

    def predict_set(self, input, p: SVPParam):
        """
        Dispatch to the appropriate set-valued prediction function.
        """
        if p.svptype == SVPType.LAC:
            if not self.root.y and p.c == self.num_classes:
                return self.lacsvp(input, p)
            elif self.root.y and p.c == self.num_classes:
                return self.lacusvphf(input, p)
            else:
                return self.lacrsvphf(input, p)
        elif p.svptype == SVPType.RAPS:
            if not self.root.y and p.c == self.num_classes:
                return self.rapssvp(input, p)
            elif self.root.y and p.c == self.num_classes:
                return self.rapsusvphf(input, p)
            else:
                return self.rapsrsvphf(input, p)
        elif p.svptype == SVPType.CSVPHF:
            if not self.root.y and p.c == self.num_classes:
                return self.rapssvp(input, p)
            elif self.root.y and p.c == self.num_classes:
                return self.rapsusvphf(input, p)
            else:
                return self.csvphfrsvphf(input, p)
        elif p.svptype == SVPType.CRSVPHF:
            if not self.root.y and p.c == self.num_classes:
                return self.rapssvp(input, p)
            elif self.root.y and p.c == self.num_classes:
                return self.rapsusvphf(input, p)
            else:
                return self.crsvphfrsvphf(input, p)
        else:
            if not self.root.y and p.c == self.num_classes:
                return self.gsvbop(input, p)
            elif not self.root.y and p.c < self.num_classes:
                return self.gsvbop_r(input, p)
            elif self.root.y and p.c == self.num_classes:
                return self.gsvbop_hf(input, p)
            else:
                return self.gsvbop_hf_r(input, p)

    # --- Simplified implementations of set-valued prediction functions ---
    def lacsvp(self, input, p: SVPParam):
        out = self.root.clf(input)
        out = F.softmax(out, dim=1)
        preds = []
        for i in range(input.size(0)):
            probs = out[i]
            sorted_idx = torch.argsort(probs, descending=True)
            cum = 0.0
            sel = []
            for idx in sorted_idx:
                cum += probs[idx].item()
                sel.append(idx.item())
                if cum >= 1 - p.error:
                    break
            preds.append(sel)
        return preds

    def lacusvphf(self, input, p: SVPParam):
        preds = []
        for i in range(input.size(0)):
            q = []
            heapq.heappush(q, QNode(self.root, 1.0))
            sel = []
            while q:
                current = heapq.heappop(q)
                if len(current.node.y) == 1:
                    if current.prob >= 1 - p.error:
                        sel.append(current.node.y[0])
                    else:
                        break
                else:
                    out = current.node.clf(input[i].view(1, -1))
                    probs = F.softmax(out, dim=1).cpu().detach().numpy()[0]
                    for j, child in enumerate(current.node.chn):
                        heapq.heappush(q, QNode(child, current.prob * probs[j]))
            preds.append(sel)
        return preds

    def lacrsvphf(self, input, p: SVPParam):
        preds = []
        for i in range(input.size(0)):
            q = []
            heapq.heappush(q, QNode(self.root, 1.0))
            ystar = None
            ystar_prime = []
            while q:
                current = heapq.heappop(q)
                if len(current.node.y) == 1:
                    if current.prob >= 1 - p.error:
                        ystar_prime.append(current.node.y[0])
                        ystar = current.node
                    else:
                        break
                else:
                    out = current.node.clf(input[i].view(1, -1))
                    probs = F.softmax(out, dim=1).cpu().detach().numpy()[0]
                    for j, child in enumerate(current.node.chn):
                        heapq.heappush(q, QNode(child, current.prob * probs[j]))
            if ystar is not None:
                # Retrieve parent via weakref.
                while (not set(ystar_prime).issubset(set(ystar.y)) and
                       ystar._meta.get('parent') is not None and
                       ystar._meta['parent']()):
                    ystar = ystar._meta['parent']()
                preds.append(ystar.y)
            else:
                preds.append([])
        return preds

    def rapssvp(self, input, p: SVPParam):
        out = self.root.clf(input)
        out = F.softmax(out, dim=1)
        preds = []
        for i in range(input.size(0)):
            sorted_idx = torch.argsort(out[i], descending=True)
            cum = 0.0
            sel = []
            for idx in sorted_idx:
                cum += out[i][idx].item()
                sel.append(idx.item())
                if cum > p.error + p.lambda_ * max(len(sel) - p.k, 0):
                    break
            preds.append(sel)
        return preds

    def rapsusvphf(self, input, p: SVPParam):
        preds = []
        for i in range(input.size(0)):
            q = []
            heapq.heappush(q, QNode(self.root, 1.0))
            sel = []
            while q:
                current = heapq.heappop(q)
                if len(current.node.y) == 1:
                    sel.append(current.node.y[0])
                    if current.prob > p.error + p.lambda_ * max(len(sel) - p.k, 0):
                        break
                else:
                    out = current.node.clf(input[i].view(1, -1))
                    probs = F.softmax(out, dim=1).cpu().detach().numpy()[0]
                    for j, child in enumerate(current.node.chn):
                        heapq.heappush(q, QNode(child, current.prob * probs[j]))
            preds.append(sel)
        return preds

    def rapsrsvphf(self, input, p: SVPParam):
        return self.rapsusvphf(input, p)

    def csvphfrsvphf(self, input, p: SVPParam):
        return self.rapsusvphf(input, p)

    def crsvphfrsvphf(self, input, p: SVPParam):
        return self.rapsusvphf(input, p)

    def gsvbop(self, input, p: SVPParam):
        out = self.root.clf(input)
        out = F.softmax(out, dim=1)
        preds = []
        for i in range(input.size(0)):
            probs = out[i]
            sorted_idx = torch.argsort(probs, descending=True)
            yhat = []
            yhat_p = 0.0
            ystar = []
            ystar_u = 0.0
            for idx in sorted_idx:
                yhat.append(idx.item())
                yhat_p += probs[idx].item()
                if p.svptype == SVPType.FB:
                    util = yhat_p * (1.0 + p.beta**2) / (len(yhat) + p.beta**2)
                elif p.svptype == SVPType.DG:
                    util = yhat_p * (p.delta / len(yhat) - p.gamma / (len(yhat)**2))
                elif p.svptype == SVPType.SIZECTRL:
                    if len(yhat) > p.size:
                        break
                    else:
                        util = yhat_p
                else:
                    if yhat_p >= 1 - p.error:
                        util = yhat_p
                    else:
                        util = 0
                if util >= ystar_u:
                    ystar = copy.copy(yhat)
                    ystar_u = util
                else:
                    break
            preds.append(ystar)
        return preds

    def gsvbop_r(self, input, p: SVPParam):
        out = self.root.clf(input)
        out = F.softmax(out, dim=1)
        if self.hstruct is not None:
            out = torch.matmul(out, self.hstruct.t())
        preds = []
        for i in range(input.size(0)):
            pred = []
            si_optimal = 0
            si_optimal_u = 0.0
            for si in range(out.size(1)):
                curr_p = out[i, si].item()
                if p.svptype == SVPType.FB:
                    size_val = self.hstruct[si].sum().item() if self.hstruct is not None else 1
                    util = curr_p * (1.0 + p.beta**2) / (size_val + p.beta**2)
                elif p.svptype == SVPType.DG:
                    size_val = self.hstruct[si].sum().item() if self.hstruct is not None else 1
                    util = curr_p * (p.delta / size_val - p.gamma / (size_val**2))
                elif p.svptype == SVPType.SIZECTRL:
                    util = curr_p
                else:
                    if curr_p >= 1 - p.error:
                        util = 1.0 / (self.hstruct[si].sum().item() if self.hstruct is not None else 1)
                    else:
                        util = 0
                if util >= si_optimal_u:
                    si_optimal = si
                    si_optimal_u = util
            if self.hstruct is not None:
                pred = [j for j in range(self.hstruct.size(1)) if self.hstruct[si_optimal, j].item() == 1]
            preds.append(pred)
        return preds

    def gsvbop_hf(self, input, p: SVPParam):
        preds = []
        for i in range(input.size(0)):
            yhat = []
            yhat_p = 0.0
            ystar = []
            ystar_u = 0.0
            q = []
            heapq.heappush(q, QNode(self.root, 1.0))
            while q:
                current = heapq.heappop(q)
                if len(current.node.y) == 1:
                    yhat.append(current.node.y[0])
                    yhat_p += current.prob
                    if p.svptype == SVPType.FB:
                        util = yhat_p * (1.0 + p.beta**2) / (len(yhat) + p.beta**2)
                    elif p.svptype == SVPType.DG:
                        util = yhat_p * (p.delta / len(yhat) - p.gamma / (len(yhat)**2))
                    elif p.svptype == SVPType.SIZECTRL:
                        if len(yhat) >= p.size:
                            break
                        else:
                            util = yhat_p
                    else:
                        if yhat_p >= 1 - p.error:
                            util = yhat_p
                        else:
                            util = 0
                    if util >= ystar_u:
                        ystar = copy.copy(yhat)
                        ystar_u = util
                    else:
                        break
                else:
                    out = current.node.clf(input[i].view(1, -1))
                    probs = F.softmax(out, dim=1).cpu().detach().numpy()[0]
                    for j, child in enumerate(current.node.chn):
                        heapq.heappush(q, QNode(child, current.prob * probs[j]))
            preds.append(ystar)
        return preds

    def gsvbop_hf_r(self, input, p: SVPParam):
        preds = []
        for i in range(input.size(0)):
            q = []
            heapq.heappush(q, QNode(self.root, 1.0))
            ystar, ystar_u = self._gsvbop_hf_r(input[i].view(1, -1), p, p.c, [], 0.0, [], 0.0, q)
            preds.append(ystar)
        return preds

    def _gsvbop_hf_r(self, input, p: SVPParam, c, ystar, ystar_u, yhat, yhat_p, q):
        while q:
            ycur = copy.copy(yhat)
            ycur_p = yhat_p
            current = heapq.heappop(q)
            ycur.extend(current.node.y)
            ycur_p += current.prob
            if p.svptype == SVPType.FB:
                util = ycur_p * (1.0 + p.beta**2) / (len(ycur) + p.beta**2)
            elif p.svptype == SVPType.DG:
                util = ycur_p * (p.delta / len(ycur) - p.gamma / (len(ycur)**2))
            elif p.svptype == SVPType.SIZECTRL:
                util = ycur_p if len(ycur) <= p.size else 0
            else:
                util = 1.0 / len(ycur) if ycur_p >= 1 - p.error else 0
            if util >= ystar_u:
                ystar = copy.copy(ycur)
                ystar_u = util
            if c > 1 and len(current.node.y) > 1:
                out = current.node.clf(input)
                probs = F.softmax(out, dim=1).cpu().detach().numpy()[0]
                for j, child in enumerate(current.node.chn):
                    heapq.heappush(q, QNode(child, current.prob * probs[j]))
                ystar, ystar_u = self._gsvbop_hf_r(input, p, c - 1, ystar, ystar_u, ycur, ycur_p, q)
            else:
                break
        return ystar, ystar_u

    # Allow external update of hstruct.
    def set_hstruct(self, hstruct):
        self.hstruct = hstruct.float()
