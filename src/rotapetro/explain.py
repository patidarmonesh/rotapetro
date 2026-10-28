from typing import Dict, Any, List

class KnowledgeBase:
    """Mock Knowledge Base interface."""
    def get_property(self, mineral: str, prop: str) -> str:
        return ""

class ExplanationGenerator:
    """Explanations (§17.4): Translates properties and features into plain English justifications."""
    
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def explain(self, prediction: str, properties: Dict[str, str], features: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        """
        Build description from predicted properties.
        Add measured features.
        Add knowledge base special text.
        Add rejection reason for runner-up.
        """
        desc = []
        
        # Build description from predicted properties
        if 'colour' in properties:
            desc.append(self._describe_colour(properties['colour']))
        if 'pleochroism' in properties:
            desc.append(self._describe_pleochroism(properties['pleochroism']))
        if 'birefringence' in properties:
            desc.append(self._describe_birefringence(properties['birefringence']))
        if 'cleavage' in properties:
            desc.append(self._describe_cleavage(properties['cleavage']))
        if 'twinning' in properties:
            desc.append(self._describe_twinning(properties['twinning']))
            
        main_desc = f"{prediction} is predicted because it exhibits " + ", ".join(desc) + "."
        
        # Add measured features
        feats = []
        if 'extinction_angle' in features:
            feats.append(self._describe_extinction(features['extinction_angle']))
            
        if feats:
            main_desc += " Measured features include " + " and ".join(feats) + "."
            
        # Add KB special text if any
        kb_text = self.kb.get_property(prediction, 'special_features')
        if kb_text:
            main_desc += f" {kb_text}"
            
        # Add rejection reason for runner-up
        runner_up = evidence.get('runner_up')
        if runner_up:
            rejection = self._explain_rejection(prediction, runner_up, properties)
            if rejection:
                main_desc += f" {rejection}"
                
        return main_desc.capitalize()

    def _describe_colour(self, val: str) -> str:
        return f"a {val} colour"

    def _describe_pleochroism(self, val: str) -> str:
        if val.lower() == 'none':
            return "no pleochroism"
        return f"{val} pleochroism"

    def _describe_birefringence(self, val: str) -> str:
        return f"{val} birefringence"

    def _describe_extinction(self, angle: float) -> str:
        if angle <= 5.0:
            return f"parallel extinction ({angle:.1f}°)"
        else:
            return f"inclined extinction ({angle:.1f}°)"

    def _describe_twinning(self, val: str) -> str:
        if val.lower() == 'none':
            return "no twinning"
        return f"{val} twinning"

    def _describe_cleavage(self, val: str) -> str:
        if val.lower() == 'none':
            return "no cleavage"
        return f"{val} cleavage"

    def _explain_rejection(self, main_pred: str, runner_up: str, props: Dict[str, str]) -> str:
        """Compare properties to explain why the runner up was rejected."""
        # Simple rule-based mock
        # E.g. "Hornblende less likely: no second cleavage"
        
        if runner_up.lower() == 'hornblende' and props.get('cleavage') == 'single':
            return f"{runner_up} less likely: requires two cleavage planes."
            
        if runner_up.lower() == 'biotite' and props.get('pleochroism') == 'none':
            return f"{runner_up} less likely: exhibits strong pleochroism."
            
        return f"{runner_up} was less likely given the observed features."
