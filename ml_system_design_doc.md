# High-Level System Design

Based on [https://github.com/eugeneyan/ml-design-docs](https://github.com/eugeneyan/ml-design-docs).

## Project Description

Define the problem statement and objectives.

## High-Level Design

Diagram of the overall system architecture (data flows, components, and interactions).
Description of the ML models involved, including short description.

## Data Requirements

Specify the datasets and data updates.
Describe (high level) the data preprocessing steps and how data will be managed and stored.

## Service Decomposition

Break down the system into microservices.
Describe the role and functionality of each service in the context of the overall system.
Explain the communication between services (e.g., REST APIs, message queues).

## Requirements Specification

List the operational requirements for each service, like: scalability, performance.
Specify the tools and technologies that you'll use for the implementation.

## Evaluation Metrics

Define how the performance of the machine learning model will be measured.
Outline the criteria for success of the overall system from both a technical and business perspective.

# ml-design-doc

## 1. Overview

> A summary of the doc's purpose, problem, solution, and desired outcome, usually in 3-5 sentences.

Autonomous driving car that can drive by itself.

## 2. Motivation

> Why the problem is important to solve, and why now.

- Decrease number of road fatalities.
- As a side goal - free people from driving.

## 3. Success metrics

> Usually framed as business goals, such as increased customer engagement (e.g., CTR, DAU), revenue, or reduced cost.

Statistics for road fatalities in Europe is 10000 per year (reference to graph and data). First goal is to reduce road fatalities by factor of 2, 5000 per year.

## 4. Requirements & Constraints

> Functional requirements are those that should be met to ship the project. They should be described in terms of the customer perspective and benefit. (See [this](https://eugeneyan.com/writing/ml-design-docs/#the-why-and-what-of-design-docs) for more details.)
>
> Non-functional/technical requirements are those that define system quality and how the system should be implemented. These include performance (throughput, latency, error rates), cost (infra cost, ops effort), security, data privacy, etc.
>
> Constraints can come in the form of non-functional requirements (e.g., cost below $`x` a month, p99 latency < `y`ms)

Functional requirements:

- Drive by itself in a defined road

Non-functional requirements:

- Realtime inference time
- Is able to run with X VRAM
- Code Quality
- Functional safety

Constraints:

- ???

### 4.1 What's in-scope & out-of-scope?

> Some problems are too big to solve all at once. Be clear about what's out of scope.

In scope:

- Vision object detection module for moving objects

Out of scope:

- Vision object detection for static objects
- Model speedup (inference time optimizations)
- VRAM constraints
- Interface with other modules
- Other modules needed for autonomous driving architecture
- Data privacy
- Data security

## 5. Methodology

### 5.1. Problem statement

> How will you frame the problem? For example, fraud detection can be framed as an unsupervised (outlier detection, graph cluster) or supervised problem (e.g., classification).

Object detection, supervised. For every frame output list of object bounding boxes, classified.

### 5.2. Data

> What data will you use to train your model? What input data is needed during serving?

ZOD dataset (link, reference)
Training: train dataset with frames with ground truth bounding boxes + corresponding lidar frames (for determining the distance to the car)
Testing: val dataset with frames.

### 5.3. Techniques

> What machine learning techniques will you use? How will you clean and prepare the data (e.g., excluding outliers) and create features?

Yolo or other state of the arch object detector. Train with data.

### 5.4. Experimentation & Validation

> How will you validate your approach offline? What offline evaluation metrics will you use?
>
> If you're A/B testing, how will you assign treatment and control (e.g., customer vs. session-based) and what metrics will you measure? What are the success and [guardrail](https://medium.com/airbnb-engineering/designing-experimentation-guardrails-ed6a976ec669) metrics?

Describe metrics here.

## 6. Implementation

### 6.1. High-level design

> ![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Data-flow-diagram-example.svg/1280px-Data-flow-diagram-example.svg.png)
> 
> Start by providing a big-picture view. [System-context diagrams](https://en.wikipedia.org/wiki/System_context_diagram) and [data-flow diagrams](https://en.wikipedia.org/wiki/Data-flow_diagram) work well.d

Draw context diagram here: reduce road fatalities -> do not crash moving objects -> avoid moving objects -> behaivior prediction for moving objects -> trajectory prediction for moving objects -> vision object detection 

For the following description propose the diagram type and write code for drawing diagram.
context: autonomous car system design. It is actually a safety goal decomposition. The goal is to show where in the architecture the vision object detection module is and what other modules will use it. Missing options (where stated "one of the options") display with a separate leaf on the save level (with ... text?)
reduce road fatalities -> (one of the options)  do not crash moving objects -> (one of the options) avoid moving objects -> (one of the options) behavior prediction for moving objects -> (one of the options) trajectory prediction for moving objects -> (one of the options) object detection  -> (consists of two parts) - vision object detection, lidar object detection

```mermaid
graph TD
    A[Reduce road fatalities]
    A --> B[...]
    A --> C[Do not crash moving objects]
    C --> D[...]
    C --> E[Avoid moving objects]
    E --> F[...]
    E --> G[Behavior prediction for moving objects]
    G --> H[...]
    G --> I[Trajectory prediction for moving objects]
    I --> J[...]
    I --> K[Object detection]
    K --> N[Vision object detection]
    K --> O[Lidar object detection]
```

### 6.2. Infra

> How will you host your system? On-premise, cloud, or hybrid? This will define the rest of this section

### 6.3. Performance (Throughput, Latency)

> How will your system meet the throughput and latency requirements? Will it scale vertically or horizontally?

Realtime, X ms for frame, Y MB VRAM peak consumption. The model shall be optimized for inference time by industry-standard inference-time optimization techniques (pruning, transfer learning, quantization). Out of scope for this design document,

### 6.8. Integration points

How will your system integrate with upstream data and downstream users?

### 6.9. Risks & Uncertainties

> Risks are the known unknowns; uncertainties are the unknown unknows. What worries you and you would like others to review?

Risks:

- Missing dynamic object may result in crash
- Misdetection un existing objects may result in dangerous maneuvers (that are better to avoid).

Uncertainties:

- Critical weather and earth conditions (for example, hurricane, tsunami, earthquake).

## 7. Appendix

### 7.1. Alternatives

What alternatives did you consider and exclude? List pros and cons of each alternative and the rationale for your decision.

### 7.2. Experiment Results

Share any results of offline experiments that you conducted.

### 7.3. Performance benchmarks

Share any performance benchmarks you ran (e.g., throughput vs. latency vs. instance size/count).

### 7.4. Milestones & Timeline

What are the key milestones for this system and the estimated timeline?

### 7.5. Glossary

Define and link to business or technical terms.

### 7.6. References

Add references that you might have consulted for your methodology.

---
## Other templates, examples, etc
- [A Software Design Doc](https://www.industrialempathy.com/posts/design-doc-a-design-doc/) `Google`
- [Design Docs at Google](https://www.industrialempathy.com/posts/design-docs-at-google/) `Google`
- [Product Spec of Emoji Reactions on Twitter Messages](https://docs.google.com/document/d/1sUX-sm5qZ474PCQQUpvdi3lvvmWPluqHOyfXz3xKL2M/edit#heading=h.554u12gw2xpd) `Twitter`
- [Design Docs, Markdown, and Git](https://caitiem.com/2020/03/29/design-docs-markdown-and-git/) `Microsoft`
- [Technical Decision-Making and Alignment in a Remote Culture](https://multithreaded.stitchfix.com/blog/2020/12/07/remote-decision-making/) `Stitchfix`
- [Design Documents for Chromium](https://www.chromium.org/developers/design-documents) `Chromium`
- [PRD Template](https://works.hashicorp.com/articles/prd-template) and [RFC Template](https://works.hashicorp.com/articles/rfc-template) (example RFC: [Manager Charter](https://works.hashicorp.com/articles/manager-charter)) `HashiCorp`
- [Pitch for To-Do Groups and Group Notifications](https://basecamp.com/shapeup/1.5-chapter-06#examples) `Basecamp`
- [The Anatomy of a 6-pager](https://writingcooperative.com/the-anatomy-of-an-amazon-6-pager-fc79f31a41c9) and an [example](https://docs.google.com/document/d/1LPh1LWx1z67YFo67DENYUGBaoKk39dtX7rWAeQHXzhg/edit) `Amazon`
- [Writing for Distributed Teams](http://veekaybee.github.io/2021/07/17/p2s/), [How P2 Changed Automattic](https://ma.tt/2009/05/how-p2-changed-automattic/) `Automattic`
- [Writing Technical Design Docs](https://medium.com/machine-words/writing-technical-design-docs-71f446e42f2e), [Writing Technical Design Docs, Revisited](https://medium.com/machine-words/writing-technical-design-docs-revisited-850d36570ec) `AWS`
- [How to write a good software design doc](https://www.freecodecamp.org/news/how-to-write-a-good-software-design-document-66fcf019569c/) `Plaid`

Contributions [welcome](https://github.com/eugeneyan/ml-design-docs/pulls)!