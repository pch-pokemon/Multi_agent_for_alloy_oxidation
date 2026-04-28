# High-Temperature Alloy Oxidation Multi-Agent System V1.0

This software is a knowledge-guided multi-agent system designed for intelligent prediction, mechanistic interpretation, thermodynamic analysis, and literature-grounded reasoning of oxidation behavior in high-temperature superalloys.

Unlike conventional tool-based frameworks, the system organizes the oxidation research process into a structured multi-agent workflow, enabling the transformation of fragmented mechanistic knowledge into design-relevant insights.

## System Architecture

The system is coordinated by a central Supervisor and consists of five specialized agents:

### (1) Machine Learning Agent (ML_agent)
- Predicts oxidation mass gain (mg/cm²) and kinetic parameters
- Supports single prediction, batch prediction, and CSV-based evaluation
- Provides composition-resolved oxidation trends

### (2) Thermodynamic Agent (Thermo_agent)
- Provides thermodynamic properties of oxides (ΔG, pO₂, equilibrium constants)
- Supports interpolation, extrapolation, and compositional scanning
- Ensures physical consistency of predicted oxidation behavior

### (3) RAG Knowledge Agent (RAG_agent)
- Retrieves and aggregates literature evidence related to oxidation mechanisms
- Supports multi-modal outputs (figures, tables, references)
- Maintains traceable evidence linkage

### (4) Mechanistic Analysis Module (logical agent, implemented within RAG_agent)
- Extracts and organizes mechanistic knowledge from literature
- Identifies element roles, interactions, and inconsistencies
- Bridges fragmented literature into interpretable knowledge structures
- Enables hypothesis generation (e.g., compositional boundaries)

### (5) Reference Agent (Ref_agent)
- Recommends relevant scientific literature
- Outputs titles, key findings, and DOI links
- Ensures reliability using curated local database

## Coordination Strategy

The Supervisor dynamically determines which agents to invoke based on user queries and task requirements. 

It does not perform inference itself, but orchestrates the workflow to:
- combine predictions
- validate thermodynamics
- retrieve supporting evidence
- construct mechanistic explanations

## Application Scenarios

- Oxidation behavior prediction
- Mechanism interpretation and hypothesis generation
- Thermodynamic analysis of alloy systems
- Literature-supported materials design


高温合金氧化行为多智能体知识问答与预测系统V1.0

本软件为高温合金氧化行为多智能体知识问答与预测系统，其核心目标是针对高温合金的氧化行为提供智能化的预测、机理解释、热力学数据查询及文献推荐支持。
系统基于多智能体架构，由协调者（Supervisor）统一调度四个子智能体：

#（1）机器学习智能体（ML_agent）
功能：基于训练好的模型，预测不同合金成分及热处理条件下的氧化增重（mg/cm²）。
支持单条预测、批量预测、CSV文件预测与评估。

#（2）热力学智能体（Thermo_agent）
功能：提供氧化物的热力学数据（吉布斯自由能 ΔG、平衡氧分压pO2、平衡常数K 等）。
支持精确查询、模糊匹配、温度插值/外推、范围曲线数据生成，以及批量查询。

#（3）RAG知识问答智能体（RAG_agent）
功能：基于检索增强生成（RAG），回答与高温合金氧化机理相关的问题。
支持返回多模态结果，包括图表链接、参考文献，并保持原始引用标记。

#（4）文献推荐智能体（Ref_agent）
功能：推荐相关学术文献，输出论文题目、关键发现及 DOI 链接。
保证结果来自本地文献数据库，并自动过滤非法引用。

架构特点：
协调调度：Supervisor 可根据用户问题自动判断调用哪一个或多个子智能体，直至解决问题并组合结果形成最终答案返回给用户。

应用场景：
科研及企业人员可利用本系统进行氧化行为预测、机理解释、热力学分析，并快速获取相关文献支撑，从而实现数据驱动的材料研究与设计优化。
