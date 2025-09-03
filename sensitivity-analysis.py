#!/usr/bin/env python3
"""
Parametric Sensitivity Analysis for AI vs Human Environmental Impact Assessment
Based on corrected methodology from Case Study analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import pprint
warnings.filterwarnings('ignore')

class AIHumanImpactCalculator:
    """Calculator for AI vs Human environmental impact with parametric analysis"""
    
    def __init__(self):
        # Baseline parameters
        self.baseline_params = {
            # High-impact parameters
            'prompts_per_task': 4,           # Number of prompts per writing task
            'gpu_utilization': 0.70,         # GPU utilization rate (70%)
            'queries_per_month': 3e9,        # Total queries per month (3 billion)
            'training_days': 3.5,            # Training duration in days
            'efficiency_gain': 0.3,          # Fraction of human time saved by the AI
            
            # Medium-impact parameters
            # 'carbon_intensity_elec': 400,         # gCO2e/kWh, IEA projected 2027 global average
            'carbon_intensity_elec': 162,    # gCO2e/kWh - 2023 UK average carbon intensity
            'carbon_intensity_gas': 185,     # gCO2e/kWh - Natural gaz carbon intensity
            'gpu_lifespan_years': 1.5,       # GPU lifespan in years, from Tomlinson et al
            'server_lifespan_years': 4,      # Server lifespan in years, Standard enterprise lifecycle
            'pue': 1.15,                     # Power Usage Effectiveness, AWS-level efficiency
            
            # Low-impact parameters (fixed for this analysis)
            'writing_time_hours': 0.83,      # Time to write one page (hours)
            'gpu_power_w': 300,              # GPU power consumption (Watts)
            'num_training_gpus': 10000,      # Number of GPUs for training
            'num_inference_gpus': 14468,     # Number of GPUs for inference
            'gpu_embodied_kg': 150,          # Embodied carbon per GPU (kg CO2e)
            'server_embodied_kg': 9180,      # Embodied carbon per server (kg CO2e)
            'gpus_per_server': 4,            # GPUs per server
            # 'num_training_servers': 2500,    # Calculated from previous parameters
            # 'num_inference_servers': 3617,   # Calculated from previous parameters
            
            # Human emissions (consistent across scenarios)
            # 'lighting_gco2e': 1.13,          # gCO2e per task
            # 'heating_gco2e': 16.63,          # gCO2e per task
            # 'laptop_operational_gco2e': 0.28, # gCO2e per task
            # 'laptop_embodied_gco2e': 4.66,    # gCO2e per task
            'annual_elec_household' : 3941,     # kWh
            'annual_gas_household' : 11500,     # kWh
            'fraction_elec_lighting' : 0.15,    # none
            'fraction_gas_heating' : 0.66,      # none
            'fraction_house_writing' : 0.125,   # none
            'laptop_daily_energy': 0.05,        # kWh
            'laptop_embodied_kg': 197,          # kgCO2e
            'laptop_lifetime_years': 4          # years

        }

        include_heating_lighting = True        
        if not(include_heating_lighting):
            self.baseline_params['annual_elec_household'] = 0
            self.baseline_params['annual_gas_household'] = 0
            self.baseline_params['pue'] = 1
        
        # Parameter ranges for sensitivity analysis
        self.param_ranges = {
            # High-impact parameters
            'prompts_per_task': [1, 4, 15],           # Right-skewed user behavior
            'gpu_utilization': [0.40, 0.70, 0.85],         # Patel et al. 2024 range
            'queries_per_month': [1e9, 3e9, 12e9],         # OpenAI 100M+ users, usage uncertainty
            'training_days': [2, 3.5, 14],              # Narayanan et al. scaling with uncertainty
            
            # Medium-impact parameters  
            'carbon_intensity_elec': [150, 400, 715],           # Denmark (clean) to India (coal-heavy)
            'gpu_lifespan_years': [1, 1.5, 2.0],        # Tech obsolescence vs durability
            'server_lifespan_years': [3, 4, 5],           # Standard enterprise cycles
            'pue': [1.08, 1.15, 1.25],                    # Google/Meta best-in-class to moderate efficiency
            'efficiency_gain': [0.8, 0.5, 0],                    # Fraction of human time saved by the AI
        }
    
    def calculate_ai_emissions(self, params: Dict) -> Dict[str, float]:
        """Calculate AI emissions per task based on parameters"""
        
        num_training_servers = params['num_training_gpus'] / params['gpus_per_server']
        num_inference_servers = params['num_inference_gpus'] / params['gpus_per_server']

        # Training emissions calculation
        training_hours = params['training_days'] * 24
        training_gpu_hours = params['num_training_gpus'] * training_hours
        training_operational_kwh = (  params['gpu_utilization'] 
                                    * params['gpu_power_w'] / 1000  # Convert W to kW
                                    * training_gpu_hours
                                    * params['pue']
                                    ) 
        training_operational_co2e = training_operational_kwh * params['carbon_intensity_elec']
        
        # Training embodied emissions
        training_gpu_embodied = (  params['num_training_gpus'] 
                                 * params['gpu_embodied_kg'] * 1000 # Convert kg to g 
                                 * training_hours 
                                 / (params['gpu_lifespan_years'] * 365 * 24))
        
        training_servers_embodied = (  num_training_servers
                                     * params['server_embodied_kg'] * 1000 # Convert kg to g 
                                     * training_hours 
                                     / (params['server_lifespan_years'] * 365 * 24))

        training_embodied = training_gpu_embodied + training_servers_embodied
        

        # Inference emissions calculation (per month)
        inference_hours_per_month = 24 * 30  # Assume 24/7 operation
        inference_operational_kwh = (  params['gpu_utilization'] 
                                     * params['gpu_power_w'] / 1000 # Convert W to kW
                                     * params['num_inference_gpus'] 
                                     * inference_hours_per_month 
                                     * params['pue']) 
        inference_operational_co2e = inference_operational_kwh * params['carbon_intensity_elec']

        # Inference embodied emissions (per month)
        inference_gpu_embodied = ( params['num_inference_gpus'] 
                                 * params['gpu_embodied_kg'] * 1000 # Convert kg to g 
                                 * inference_hours_per_month 
                                 / (params['gpu_lifespan_years'] * 365 * 24))
        inference_server_embodied = (  num_inference_servers 
                                     * params['server_embodied_kg'] * 1000 # Convert kg to g 
                                     * inference_hours_per_month 
                                     / (params['server_lifespan_years'] * 365 * 24))
        inference_embodied = inference_gpu_embodied + inference_server_embodied


        # Per-query emissions
        inference_operational_kwh_per_query = inference_operational_kwh / params['queries_per_month']
        training_operational_per_query = training_operational_co2e / params['queries_per_month']
        training_embodied_per_query = training_embodied / params['queries_per_month']
        inference_operational_per_query = inference_operational_co2e / params['queries_per_month']
        inference_embodied_per_query = inference_embodied / params['queries_per_month']
        
        # Per-task emissions (multiply by prompts per task)
        ai_emissions = {
            'c_training_operational': training_operational_per_query * params['prompts_per_task'],
            'c_training_embodied': training_embodied_per_query * params['prompts_per_task'],
            'c_inference_operational': inference_operational_per_query * params['prompts_per_task'],
            'c_inference_embodied': inference_embodied_per_query * params['prompts_per_task'],
        }

        ai_emissions['total'] = sum(ai_emissions.values())
        ai_emissions['inference_energy'] = inference_operational_kwh_per_query

        return ai_emissions
    
    def calculate_human_emissions(self, params: Dict) -> Dict[str, float]:
        """Calculate human emissions per task"""
        c_lighting = (params['annual_elec_household']*params['fraction_elec_lighting']
                      *params['fraction_house_writing']*params['writing_time_hours']/(24*365)
                      *params['carbon_intensity_elec'])
        c_heating = (params['annual_gas_household']*params['fraction_gas_heating']
                      *params['fraction_house_writing']*params['writing_time_hours']/(24*365)
                      *params['carbon_intensity_gas'])
        c_laptop_ope = params['laptop_daily_energy']*params['writing_time_hours']/24*params['carbon_intensity_elec']
        c_laptop_emb = params['laptop_embodied_kg']*1000*params['writing_time_hours']/(params['laptop_lifetime_years']*24*356)

        human_emissions = {
            'c_lighting':c_lighting,
            'c_heating':c_heating,
            'c_laptop_ope':c_laptop_ope,
            'c_laptop_emb':c_laptop_emb}
        human_emissions['total'] = sum(human_emissions.values())
        
        return human_emissions
    
    def calculate_net_impact(self, params: Dict) -> Tuple[float, Dict]:
        """Calculate net impact (AI - Human emissions)"""
        ai_emissions = self.calculate_ai_emissions(params)
        human_emissions = self.calculate_human_emissions(params)
        
        # A fraction of the human emissions may be avoided in the solution scenario due to time eficiency gain from the AI
        net_impact = ai_emissions['total'] - human_emissions['total'] * params['efficiency_gain']
        
        return net_impact, {
            'inference_energy': ai_emissions['inference_energy'],
            'ai_total': ai_emissions['total'],
            'ai_breakdown': ai_emissions,
            'human_breakdown': human_emissions,
            'ai_+_human': ai_emissions['total']+human_emissions['total'], # That's weird, means nothing
            # ai+human is essentially the net impact!
            'human_total': human_emissions['total'],
            'net_impact': net_impact
        }

    def sensitivity_analysis(self) -> pd.DataFrame:
        """Perform one-at-a-time sensitivity analysis"""
        results = []

        # Calculate baseline
        baseline_net_impact, baseline_details = self.calculate_net_impact(self.baseline_params)
        
        for param_name, param_values in self.param_ranges.items():
            for param_value in param_values:
                # Create modified parameters
                modified_params = self.baseline_params.copy()
                modified_params[param_name] = param_value
                
                # Calculate impact
                net_impact, details = self.calculate_net_impact(modified_params)
                
                results.append({
                    'parameter': param_name,
                    'value': param_value,
                    'net_impact': net_impact,
                    'ai_total': details['ai_total'],
                    'human_total': details['human_total'],
                    'delta_from_baseline': net_impact - baseline_net_impact,
                    'is_baseline': param_value == self.baseline_params[param_name]
                })
        
        return pd.DataFrame(results)
    
    def scenario_analysis(self) -> pd.DataFrame:
        """Perform scenario-based analysis"""
        scenarios = {
            'Optimistic': {
                # High-impact parameters (favor AI)
                'prompts_per_task': 1,              # Perfect first attempt
                'gpu_utilization': 0.40,            # Low utilization (mixed cloud workloads)
                'queries_per_month': 12e9,          # High query volume (better amortization of training)
                'training_days': 2,                 # Highly optimized training
                
                # Medium-impact parameters (best efficiency)
                'carbon_intensity_elec': 150,            # Clean grid (Denmark-level)
                'gpu_lifespan_years': 2.0,          # Extended hardware life
                'server_lifespan_years': 5,         # Extended server life
                'pue': 1.08,  
            },
            'Realistic': self.baseline_params,
            'Pessimistic': {
                # High-impact parameters (unfavorable to AI)
                'prompts_per_task': 15,             # Heavy iteration for complex tasks
                'gpu_utilization': 0.85,            # High utilization (dedicated ML workloads)
                'queries_per_month': 1e9,           # Low query volume (poor amortization of training)
                'training_days': 14,                # Extended training with inefficiencies
                
                # Medium-impact parameters (worst efficiency)
                'carbon_intensity_elec': 715,            # Coal-heavy grid (India-level)
                'gpu_lifespan_years': 1,          # Rapid technological obsolescence
                'server_lifespan_years': 3,         # Aggressive replacement cycle
                'pue': 1.25,                        # Moderately efficient data center
            }
        }
        
        results = []
        for scenario_name, scenario_params in scenarios.items():
            # Merge scenario params with baseline (scenario params override baseline)
            full_params = self.baseline_params.copy()
            full_params.update(scenario_params)
            
            net_impact, details = self.calculate_net_impact(full_params)
            
            results.append({
                'scenario': scenario_name,
                'net_impact': net_impact,
                'ai_total': details['ai_total'],
                'human_total': details['human_total'],
                'ai_training_operational': details['ai_breakdown']['c_training_operational'],
                'ai_training_embodied': details['ai_breakdown']['c_training_embodied'],
                'ai_inference_operational': details['ai_breakdown']['c_inference_operational'],
                'ai_inference_embodied': details['ai_breakdown']['c_inference_embodied']
            })
        
        return pd.DataFrame(results)
    
    
    def create_tornado_diagram(self, sensitivity_df: pd.DataFrame):
        """Create tornado diagram showing parameter sensitivity"""
        # Calculate ranges for each parameter
        param_ranges = []
        for param in self.param_ranges.keys():
            param_data = sensitivity_df[sensitivity_df['parameter'] == param]
            min_impact = param_data['net_impact'].min()
            max_impact = param_data['net_impact'].max()
            range_size = max_impact - min_impact
            
            param_ranges.append({
                'parameter': param,
                'min_impact': min_impact,
                'max_impact': max_impact,
                'range_size': range_size
            })
        
        # Sort by range size
        param_ranges = sorted(param_ranges, key=lambda x: x['range_size'], reverse=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(param_ranges))
        
        for i, param_data in enumerate(param_ranges):
            min_val = param_data['min_impact']
            max_val = param_data['max_impact']
            
            # Plot horizontal bar
            ax.barh(i, max_val - min_val, left=min_val, height=0.6, 
                   color='steelblue', alpha=0.7)
            
            # Add parameter name
            param_name = param_data['parameter'].replace('_', ' ').title()
            ax.text(-0.8, i, param_name, ha='right', va='center')
            
            # Add range values
            ax.text(max_val + 0.03, i, f"±{param_data['range_size']:.2f}", 
                   ha='left', va='center', fontsize=10)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Net Impact (gCO2e/task)')
        # ax.set_ylabel(self.param_ranges.keys())
        ax.set_title('Net impact (gCO2e) ranges for each parameter')
        ax.set_yticks([])
        # ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./tornado_plot.png')
        return fig
    
    def run_full_analysis(self):
        """Run complete sensitivity and scenario analysis"""
        print("Running Parametric Sensitivity Analysis...")
        print("=" * 50)
        
        # Sensitivity analysis
        sensitivity_df = self.sensitivity_analysis()
        
        # Scenario analysis
        scenario_df = self.scenario_analysis()

        
        # Print results
        print("\n1. BASELINE RESULTS")
        baseline_net_impact, baseline_details = self.calculate_net_impact(self.baseline_params)
        print(f"Net Impact: \t {baseline_net_impact:.2f} gCO2e/task")
        print(f"AI only: \t {baseline_details['ai_total']:.2f} gCO2e/task")
        print(f"Human only: \t {baseline_details['human_total']:.2f} gCO2e/task")
        pprint.pprint(baseline_details['ai_breakdown'])
        pprint.pprint(baseline_details['human_breakdown'])
        # print(f"AI + human: {baseline_details['ai_+_human']:.2f} gCO2e/task")
        # print("Operational energy of inference per prompt", baseline_details['inference_energy'])
        
        print("\n2. PARAMETER SENSITIVITY RANGES")
        print("the range of net impact in gCO2 that results from varying this param off the baseline")
        for param in self.param_ranges.keys():
            param_data = sensitivity_df[sensitivity_df['parameter'] == param]
            min_impact = param_data['net_impact'].min()
            max_impact = param_data['net_impact'].max()
            range_size = max_impact - min_impact
            print(f"{param:20s}: [{min_impact:6.2f}, {max_impact:6.2f}] (±{range_size:5.2f})")
        
        print("\n3. SCENARIO ANALYSIS")
        for _, row in scenario_df.iterrows():
            print(f"{row['scenario']:12s}: {row['net_impact']:6.2f} gCO2e/task")
        
        
        # Create tornado diagram
        tornado_fig = self.create_tornado_diagram(sensitivity_df)
        
        return {
            'sensitivity_df': sensitivity_df,
            'scenario_df': scenario_df,
            'tornado_fig': tornado_fig,
            'baseline_results': baseline_details
        }

if __name__ == "__main__":
    # Run the analysis
    calculator = AIHumanImpactCalculator()
    results = calculator.run_full_analysis()
    
    # Show the tornado diagram
    plt.show()
    
    # Save results to CSV
    results['sensitivity_df'].to_csv('sensitivity_analysis.csv', index=False)
    results['scenario_df'].to_csv('scenario_analysis.csv', index=False)
    
    print("\nAnalysis complete! Results saved to CSV files.")
    print("Tornado diagram saved to project root.")
