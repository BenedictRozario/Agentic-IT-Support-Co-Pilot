# src/agentic/core_agent.py

from typing import List, Dict, Any, Optional
import json
import traceback
import re


class SimpleAgent:
    def __init__(self, tools, embedder, max_cycles: int = 2, llm=None):
        """
        tools: Tools() wrapper instance providing search, create_ticket, etc.
        embedder: embedder instance for RAG
        max_cycles: maximum thought/action cycles per query
        llm: optional LLM wrapper (e.g., OllamaLLM) to polish final answer wording
        """
        self.tools = tools
        self.embedder = embedder
        self.max_cycles = max_cycles
        self.llm = llm

        # internal state / tracking
        self.last_ticket = None  # (optional) store last created ticket


    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def reason_and_act(self, user_query: str) -> Dict[str, Any]:
        """
        Main agent loop. Returns a dict with:
        - 'answer': final human-friendly message
        - 'history': per-cycle trace (plan, tools, reflection)
        - 'diagnostics': retrieval/tool/reflection logs
        """
        original_query = user_query.strip()
        history: List[Dict[str, Any]] = []
        diagnostics = {"retrieval": [], "plans": [], "tool_calls": [], "reflections": []}

        # 0) Check if this is a follow-up like "not fixed INC0010036" / "fixed INC0010036"
        followup = self._detect_followup_intent(original_query)
        if followup:
            answer = self._handle_followup(followup)
            return {
                "answer": answer,
                "history": history,
                "diagnostics": diagnostics,
            }

        # 1) Pre-parse logs (best-effort)
        parsed = {}
        try:
            if hasattr(self.tools, "parse_log_tool"):
                parsed = self.tools.parse_log_tool(original_query)
        except Exception:
            parsed = {}

        current_query = original_query
        created_ticket: Optional[Dict[str, Any]] = None
        last_fix_result: Optional[Dict[str, Any]] = None
        last_retrieval: Dict[str, Any] = {}

        for cycle in range(1, self.max_cycles + 1):
            # 2) Retrieve grounding
            try:
                retrieval = self.tools.search_local(current_query, k=4)
            except Exception:
                retrieval = {"results": []}
            diagnostics["retrieval"].append(retrieval)
            last_retrieval = retrieval

            # 3) Plan
            plan = self._make_plan(current_query, retrieval, parsed)
            diagnostics["plans"].append(plan)

            # 4) Execute plan
            tool_outcomes: List[Dict[str, Any]] = []
            for step in plan.get("steps", []):
                stype = step.get("type")
                t_out = {"step": step, "out": None}
                try:
                    if stype == "parse_result":
                        t_out["out"] = parsed

                    elif stype == "search":
                        q = step.get("query", current_query)
                        t_out["out"] = self.tools.search_local(q, k=step.get("k", 3))

                    elif stype == "create_ticket":
                        short = step.get("short") or f"{step.get('code','Issue')} - Auto-ticket"
                        desc = step.get("desc") or parsed.get("last_error_line", current_query[:1000])
                        grp = step.get("assign_group")
                        created = self.tools.create_ticket(short, desc, group=grp)
                        t_out["out"] = created
                        if isinstance(created, dict) and "error" not in created:
                            created_ticket = created
                            self.last_ticket = created  # remember last ticket
                            # optional audit
                            try:
                                from .audit import audit_log
                                audit_log(
                                    {
                                        "action": "create_ticket",
                                        "sys_id": created.get("sys_id"),
                                        "number": created.get("number"),
                                        "short": created.get("short_description"),
                                    }
                                )
                            except Exception:
                                pass

                    elif stype == "perform_fix":
                        candidate_cmd = self._derive_fix_command(step, retrieval, tool_outcomes)
                        if candidate_cmd:
                            run_real = bool(step.get("run_for_real", False))
                            fix_res = self.tools.perform_fix(candidate_cmd, run_for_real=run_real)
                            t_out["out"] = fix_res
                            last_fix_result = fix_res
                        else:
                            t_out["out"] = {
                                "skipped": True,
                                "reason": "no_actionable_command_found",
                            }

                    elif stype == "compute":
                        if hasattr(self.tools, "compute"):
                            t_out["out"] = self.tools.compute(step.get("expr", ""))
                        else:
                            t_out["out"] = {"error": "compute_tool_not_available"}

                    else:
                        t_out["out"] = {"error": f"unknown_step_type:{stype}"}

                except Exception as e:
                    t_out["out"] = {
                        "error": "tool_execution_exception",
                        "message": str(e),
                        "trace": traceback.format_exc(),
                    }

                tool_outcomes.append(t_out)

            diagnostics["tool_calls"].append(tool_outcomes)

            # 5) Post-fix ticket updates
            self._post_fix_ticket_updates(created_ticket, last_fix_result)

            # 6) Reflection
            reflection = self._reflect(
                original_query=original_query,
                retrieval=last_retrieval,
                parsed=parsed,
                created_ticket=created_ticket,
                fix_result=last_fix_result,
            )
            diagnostics["reflections"].append(reflection)

            # 7) Synthesis
            final_answer = self._synthesize_answer(
                original_query=original_query,
                retrieval=last_retrieval,
                parsed=parsed,
                created_ticket=created_ticket,
                fix_result=last_fix_result,
                reflection=reflection,
            )
            # Let Ollama rewrite the response (optional enhancement)
            if self.llm is not None:
                try:
                    final_answer = self.llm.rewrite_answer(original_query, final_answer)
                except Exception:
                    # Never break agent behavior if LLM is unavailable
                    pass

            history.append(
                {
                    "cycle": cycle,
                    "plan": plan,
                    "tool_outcomes": tool_outcomes,
                    "answer": final_answer,
                    "reflection": reflection,
                }
            )

            # For now, stop after one cycle
            break

        return {"answer": final_answer, "history": history, "diagnostics": diagnostics}

    # ------------------------------------------------------------------
    # Follow-up intent detection: "fixed" / "not fixed"
    # ------------------------------------------------------------------
    def _detect_followup_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detects follow-up intents like:
        - "not fixed"
        - "not fixed INC0010036"
        - "still not working INC0010036"
        - "fixed" / "resolved" / "it works now"
        Returns a dict with {'intent': 'fixed'|'not_fixed', 'ticket_number': Optional[str]}
        """
        t = text.strip().lower()
        # ticket number pattern
        m = re.search(r"\binc\d{7}\b", t, re.IGNORECASE)
        ticket_number = m.group(0).upper() if m else None

        if "not fixed" in t or "still not" in t or "still failing" in t or "still broken" in t:
            return {"intent": "not_fixed", "ticket_number": ticket_number}
        if "fixed" in t or "resolved" in t or "working now" in t:
            # avoid mis-classifying "not fixed"
            if "not fixed" not in t:
                return {"intent": "fixed", "ticket_number": ticket_number}
        return None

    def _handle_followup(self, followup: Dict[str, Any]) -> str:
        """
        Handle "fixed"/"not fixed" for an existing ticket:
        - Updates ServiceNow (work notes, state, escalation)
        - Does NOT create a new ticket
        - Returns a human-friendly response
        """
        intent = followup["intent"]
        ticket_no = followup.get("ticket_number")

        # need ServiceNow integration
        if not hasattr(self.tools, "sn") or not self.tools.sn:
            return (
                "I understood this as a follow-up ("
                f"{intent.replace('_', ' ')}), but I don't have ServiceNow configured, "
                "so I can't update the existing ticket."
            )

        sn = self.tools.sn

        # If no ticket number mentioned, fall back to last_ticket if available
        incident = None
        resolved_num = None
        if ticket_no:
            try:
                incident = sn.get_incident_by_number(ticket_no)
                resolved_num = ticket_no
            except Exception:
                incident = None
        elif self.last_ticket and isinstance(self.last_ticket, Dict):
            resolved_num = self.last_ticket.get("number")
            if resolved_num:
                try:
                    incident = sn.get_incident_by_number(resolved_num)
                except Exception:
                    incident = None

        if not incident:
            return (
                "I understood that the issue is "
                f"{'still not fixed' if intent == 'not_fixed' else 'fixed'}, "
                "but I couldn’t locate the corresponding incident in ServiceNow. "
                "Please mention the ticket number like `INC0010036`."
            )

        sys_id = incident.get("sys_id")
        number = incident.get("number", resolved_num or "UNKNOWN")

        try:
            if intent == "not_fixed":
                # 1) Update ServiceNow: note + escalate to Tier 2
                note = "User reports that the issue is NOT fixed yet."
                try:
                    sn.add_comment(sys_id, note, work_note=True)
                except Exception:
                    sn.update_incident(sys_id, {"comments": note})
                try:
                    gid = sn.find_group_sysid("Tier 2")
                    if gid:
                        sn.update_incident(sys_id, {"assignment_group": gid})
                except Exception:
                    pass

                # 2) Notify n8n so L2 workflow can try auto-remediation
                try:
                    if hasattr(self.tools, "notify_n8n"):
                        # pass the full incident object we already fetched
                        self.tools.notify_n8n("incident_not_fixed", incident)
                except Exception:
                    # don't break the agent if n8n call fails
                    pass

                return (
                    f"Thanks for the update. I've added a note that the issue is NOT fixed "
                    f"and escalated incident **{number}** to Tier 2 support. "
                    "Our automation engine will also attempt additional remediation."
                )


            if intent == "fixed":
                note = "User confirmed that the issue is fixed."
                try:
                    # internal work note
                    sn.add_comment(sys_id, note, work_note=True)
                except Exception:
                    # fallback to customer-visible comment
                    sn.update_incident(sys_id, {"comments": note})

                # TODO: adjust field name / value if your instance uses a different Resolution code field
                update_fields = {
                    "incident_state": "6",          # often 'Resolved'
                    "state": "6",                   # some UIs bind to 'state'
                    "close_notes": "Issue confirmed fixed by user.",
                    "close_code": "Solved Remotely",  # <- must be a valid Resolution code in *your* instance
                }

                try:
                    sn.update_incident(sys_id, update_fields)
                except Exception:
                    pass

                return (
                    f"That's great to hear! I've updated incident **{number}** as resolved "
                    "and added a note that you confirmed the fix."
                )





        except Exception as e:
            return (
                f"I recognized this as a follow-up on ticket {number}, "
                "but there was an error updating it in ServiceNow: "
                f"{str(e)}"
            )

        # Fallback
        return (
            f"I recognized this as a follow-up on ticket {number}, "
            "but I couldn't complete the update. Please check ServiceNow manually."
        )

    # ------------------------------------------------------------------
    # Planning (unchanged from previous version)
    # ------------------------------------------------------------------
    # def _make_plan(self, query: str, retrieval: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    #     steps = []
    #     codes = parsed.get("codes", []) if parsed else []

    #     if codes:
    #         steps.append({"type": "parse_result", "codes": codes, "parsed": parsed})
    #         steps.append({"type": "search", "query": codes[0], "k": 3})
    #         steps.append(
    #             {
    #                 "type": "create_ticket",
    #                 "code": codes[0],
    #                 "short": f"{codes[0]}: Auto-ticket (initial)",
    #                 "desc": "",
    #                 "assign_group": "Tier 2",
    #             }
    #         )
    #         steps.append(
    #             {
    #                 "type": "perform_fix",
    #                 "code": codes[0],
    #                 "run_for_real": False,
    #             }
    #         )
    #     else:
    #         steps.append({"type": "search", "query": query, "k": 3})
    #         steps.append(
    #             {
    #                 "type": "create_ticket",
    #                 "short": "Auto-ticket: investigation required",
    #                 "desc": query[:1000],
    #                 "assign_group": "Tier 2",
    #             }
    #         )

    #     return {"steps": steps}

    def _make_plan(self, query: str, retrieval: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
        steps = []
        codes = parsed.get("codes", []) if parsed else []

        # Use the “most useful” text as the issue summary
        issue_text = parsed.get("last_error_line") or query

        if codes:
            steps.append({"type": "parse_result", "codes": codes, "parsed": parsed})
            steps.append({"type": "search", "query": codes[0], "k": 3})
            steps.append({
                "type": "create_ticket",
                "code": codes[0],
                # short description = code + truncated issue text
                "short": f"{codes[0]}: {issue_text[:70]}",
                # full text in description
                "desc": issue_text,
                "assign_group": "Tier 2",
            })
            steps.append({
                "type": "perform_fix",
                "code": codes[0],
                "run_for_real": False,
            })
        else:
            steps.append({"type": "search", "query": query, "k": 3})
            steps.append({
                "type": "create_ticket",
                # short description directly from user text
                "short": issue_text[:80],
                "desc": issue_text,
                "assign_group": "Tier 2",
            })

        return {"steps": steps}


    # ------------------------------------------------------------------
    # Command derivation, ticket updates, synthesis, reflection
    # (same as your last working version, shortened for clarity)
    # ------------------------------------------------------------------
    def _derive_fix_command(
        self,
        step: Dict[str, Any],
        retrieval: Dict[str, Any],
        tool_outcomes: List[Dict[str, Any]],
    ) -> Optional[str]:
        if step.get("command"):
            return step["command"]

        last_search = None
        for prev in reversed(tool_outcomes):
            if prev["step"].get("type") == "search":
                last_search = prev["out"]
                break
        if last_search is None:
            last_search = retrieval

        if not last_search:
            return None

        hits = last_search.get("results", [])
        if not hits:
            return None

        top_chunk = hits[0][1].get("chunk", "").lower()

        if "cp /opt/configs/agent-config-backup.yaml" in top_chunk:
            return "cp /opt/configs/agent-config-backup.yaml /etc/agent/config.yaml"
        if "systemctl restart agent.service" in top_chunk:
            return "systemctl restart agent.service"
        if "systemctl restart auth.service" in top_chunk:
            return "systemctl restart auth.service"
        if "systemctl restart network-agent.service" in top_chunk:
            return "systemctl restart network-agent.service"

        return None

    def _post_fix_ticket_updates(
        self,
        created_ticket: Optional[Dict[str, Any]],
        fix_result: Optional[Dict[str, Any]],
    ) -> None:
        if not created_ticket or not isinstance(created_ticket, Dict):
            return
        if fix_result is None:
            return
        if not hasattr(self.tools, "sn") or not self.tools.sn:
            return

        sys_id = created_ticket.get("sys_id")
        if not sys_id:
            return

        try:
            note = f"Auto-fix attempt result: {json.dumps(fix_result, default=str)[:1500]}"
            try:
                self.tools.sn.add_comment(sys_id, note, work_note=True)
            except Exception:
                self.tools.sn.update_incident(sys_id, {"comments": note})

            if fix_result.get("ok"):
                try:
                    self.tools.sn.update_incident(
                        sys_id,
                        {
                            "incident_state": "6",
                            "close_notes": "Auto-fix applied by agent and reported as successful.",
                        },
                    )
                except Exception:
                    pass
            else:
                try:
                    gid = self.tools.sn.find_group_sysid("Tier 2")
                    if gid:
                        self.tools.sn.update_incident(sys_id, {"assignment_group": gid})
                except Exception:
                    pass

        except Exception:
            pass

    def _synthesize_answer(
        self,
        original_query: str,
        retrieval: Dict[str, Any],
        parsed: Dict[str, Any],
        created_ticket: Optional[Dict[str, Any]],
        fix_result: Optional[Dict[str, Any]],
        reflection: Dict[str, Any],
    ) -> str:
        lines: List[str] = []

        lines.append("Thanks for sharing the issue.\n")

        lines.append("📌 **Issue I see**")
        if parsed.get("codes"):
            codes_str = ", ".join(parsed["codes"])
            lines.append(f"- Detected error code(s): `{codes_str}`")
        if parsed.get("last_error_line"):
            lines.append(f"- Last error line: `{parsed['last_error_line']}`")
        else:
            lines.append(f"- Description: {original_query}")

        hits = retrieval.get("results", [])
        if hits:
            lines.append("\n📚 **What I found in our knowledge base**")
            for score, meta in hits[:3]:
                title = meta.get("title", meta.get("doc_id", ""))
                snippet = meta.get("chunk", "")[:220].replace("\n", " ")
                lines.append(f"- **{title}** (relevance {score:.2f}): {snippet}...")
        else:
            lines.append("\n📚 I couldn't find a strong KB match yet, so I'm treating this as a new issue.")

        lines.append("\n🤖 **What I’ve done for you**")
        if created_ticket and isinstance(created_ticket, Dict) and "error" not in created_ticket:
            num = created_ticket.get("number") or created_ticket.get("sys_id") or "UNKNOWN"
            lines.append(f"- Created an incident in ServiceNow for tracking: **{num}**")
        else:
            lines.append("- I attempted to create an incident, but there was an error with ServiceNow.")

        if fix_result is not None:
            if fix_result.get("skipped"):
                lines.append("- I checked for an automated fix, but did not find a safe command to run.")
            elif fix_result.get("ok"):
                if fix_result.get("simulated"):
                    lines.append("- Attempted an automated fix (simulated) based on the KB runbook.")
                else:
                    lines.append("- Attempted an automated fix on the system based on the KB runbook.")
            else:
                lines.append("- I tried an automated fix, but it appears to have failed.")

        confidence = reflection.get("confidence", 0.0)
        clarifying = reflection.get("clarifying_question")

        lines.append("\n🔍 **What you can do next**")
        if confidence >= 0.6:
            lines.append("- Please try the system again and see if the error is resolved.")
        else:
            lines.append("- I’m not fully confident this is completely resolved yet.")
            if clarifying:
                lines.append(f"- {clarifying}")

        if created_ticket and isinstance(created_ticket, Dict) and "error" not in created_ticket:
            num = created_ticket.get("number") or created_ticket.get("sys_id") or "UNKNOWN"
            lines.append(
                f"- If the problem persists, reply with **\"not fixed {num}\"** so I can escalate it, "
                "or **\"fixed {num}\"** if everything is working."
            )

        return "\n".join(lines)

    def _reflect(
        self,
        original_query: str,
        retrieval: Dict[str, Any],
        parsed: Dict[str, Any],
        created_ticket: Optional[Dict[str, Any]],
        fix_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        hits = retrieval.get("results", [])
        avg_score = 0.0
        if hits:
            try:
                avg_score = sum(float(s) for s, _ in hits) / max(1, len(hits))
            except Exception:
                avg_score = 0.0

        confidence = float(max(0.0, min(1.0, avg_score)))

        ask_more_info = confidence < 0.6
        clarifying_question = None
        if ask_more_info:
            clarifying_question = (
                "Could you share the agent version, OS details, and the last ~20 lines of the relevant log file?"
            )

        notes = (
            "Grounded in KB and auto-fix attempted" if confidence >= 0.6
            else "Grounding is weak; requested more diagnostic information"
        )

        return {
            "confidence": confidence,
            "ask_more_info": ask_more_info,
            "clarifying_question": clarifying_question,
            "notes": notes,
        }
