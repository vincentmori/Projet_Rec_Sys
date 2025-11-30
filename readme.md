# Travel Recommendation System – TripplyBuddy

## 1. Startup Team
The team is composed of Alexis Lulin, Project Lead, who oversees project planning,
repository management, and global coordination. The data pipeline and the entire
data processing workflow are handled by Augustin Mouton, Data Engineer. The
machine learning architecture is supervised by Antonin Doat, Lead ML Engineer,
who is responsible for the design and implementation of the core recommendation
model. Model training, evaluation procedures, and deployment operations are
managed by Nicolas Laine, ML Engineer. System integration, full-stack development,
and the overall application infrastructure are delivered and maintained by Vincent
Morin, Systems Engineer.

## 2. Problem and Domain

### A. Domain: Personalized Travel Recommendation
TripplyBuddy operates within the travel-technology sector and aims to redefine how
travelers discover new destinations. With the overwhelming amount of information
available online, users often struggle to identify trips aligned with their interests,
budget, personal constraints, and evolving habits. The objective of the platform is to
provide adaptive, personalized, and context-aware recommendations that reflect the
unique characteristics of each traveler.

### B. Problem Definition
Most existing travel recommendation systems rely on simple popularity metrics or
generic heuristics, producing repetitive suggestions that fail to capture the dynamic
nature of user interests. Travelers shift between different motivations such as
adventure, relaxation, cultural immersion, environmental preferences, or lifestyle
requirements. TripplyBuddy aims to overcome these limitations by offering a system
that adapts continuously to user behavior and generates suggestions that integrate
context, personal history, and travel patterns.

### C. Target Users
The system is intended for frequent travelers seeking automated trip planning,
occasional vacationers interested in thematic recommendations such as family-
friendly or romantic trips, and digital nomads searching for destinations that offer an
optimal combination of cost, infrastructure, and connectivity suited to remote work.

### D. Value Proposition
TripplyBuddy provides highly personalized travel suggestions by learning from user
behavior and adjusting to preference changes over time. The platform contributes to
sustainable tourism by encouraging authentic, culturally rich, and less crowded
destinations. Positioned at the intersection of artificial intelligence and the travel
industry, TripplyBuddy presents strong opportunities for innovation, user growth, and
scalable development.

## 3. Innovation: HGIB-Based Recommendation Model

### A. Reference Paper
This project is inspired by the paper “Hierarchical Graph Information Bottleneck for
Multi-Behavior Recommendation” by Zhang et al. (2025). The HGIB framework is
designed to preserve the most discriminative information required for predicting user
behavior while compressing redundant or irrelevant content. It combines the
Information Bottleneck principle with graph neural networks and integrates a Graph
Refinement Encoder capable of removing noisy or uninformative interactions. The
framework also uses auxiliary losses to maintain behavior-relevant information.

In this project, HGIB is adapted to the travel domain. Destination selection is treated
as the primary behavior, while accommodation type, transportation mode, and
budget level act as auxiliary behaviors that enhance the representation of travel
preferences.

### B. Context Awareness
The model incorporates contextual information such as the trip start date, trip
duration, and traveler characteristics including age and nationality. These features
serve as auxiliary contextual signals that enrich the user representation. This design
follows the same principle as HGIB’s use of multiple behavior types, allowing the
system to refine embeddings and improve destination prediction accuracy.

### C. Multi-Behavior Graph Construction
In the original HGIB framework, graphs are built for behaviors such as viewing,
adding items to cart, or purchasing. In our travel-based adaptation, separate graphs
represent accommodation preferences, transportation choices, and budget intensity.
These graphs capture distinct travel behaviors and help the hierarchical GNN model
the complex relationships between travelers, destination attributes, and trip
patterns.

### D. Explainability
The hierarchical GNN architecture produces interpretable attention weights that can
be used to justify recommendations. For example, a destination may be suggested
because the user tends to prefer longer trips, specific accommodation types, or
regions with convenient transportation options. This mechanism mirrors HGIB’s
ability to emphasize task-relevant features while suppressing noise.

### E. Hybridization
To reduce cold-start issues for new users, the system combines collaborative
embeddings derived from HGIB with content-based features extracted from
destinations, such as cultural attributes, climate categories, or geographic types. This
hybrid design makes it possible to deliver robust recommendations even when user
history is limited.

### F. Summary
The adaptation of HGIB for the travel domain is based on four major components:
contextual signals, multi-behavior graph construction, attention-based interpretability,
and hybrid representations for cold-start scenarios. Together, these mechanisms
enable a reliable and expressive travel recommendation system.

## 4. Technical Stack

### A. Libraries
The system uses standard data processing tools such as NumPy, Pandas, and
Scikit-learn. Data visualization is carried out using Matplotlib and Seaborn. The core
model is implemented using PyTorch combined with PyTorch Geometric or DGL for
graph learning. A Streamlit interface is used to present results and provide an
interactive user experience.

### B. Framework
The system is built upon the HGIB framework, which learns compact and
information-rich representations for multi-behavior interactions. The Graph
Refinement Encoder is included to remove noisy edges from the interaction graph
and minimize negative transfer between heterogeneous behaviors.

### C. Database
PostgreSQL is used as the main database because it is scalable, reliable, and
suitable for structured travel data. The Kaggle Traveler Trip dataset is loaded into a
principal table that can later be extended or normalized into multiple tables. A user
account table stores identifiers, hashed passwords, and optional metadata. Model
embeddings are stored using the pgvector extension to enable efficient similarity
search and recommendation retrieval. A Python integration layer manages all
interactions between the model, the backend, and the database.

## 5. Installation and Launch

### A. Requirements
To begin, create and activate a virtual environment.

On macOS:
python3.12 -m venv rec_sys_proj_venv
source rec_sys_proj_venv/bin/activate

On Windows:
python -m venv rec_sys_proj_venv
.\rec_sys_proj_venv\Scripts\Activate.ps1
If error:
Set-ExecutionPolicy RemoteSigned -Scope Process
then redo:
.\rec_sys_proj_venv\Scripts\Activate.ps1

### B. Installing Dependencies
After activating the environment, install the required packages:
pip install -r requirements.txt

### C. Create .env
DATABASE_URL="postgresql+psycopg2://ucbkaif701fjf1:pef019c3014489fb0d4b482b664a245b47a18c836273a1e0930491309907a99f8@cet8gijgk7sjl9.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d2i3gudil1qocc"

### D. Launching the Streamlit Application
To start the Streamlit user interface:
streamlit run Python/Frontend/app.py

## 6. Project Structure
The Data directory contains the dataset.

The Backend directory includes all scripts related to model processing, the data pipeline, and database operations. 

The Frontend directory includes the Streamlit application and its interface logic. 

The requirements file contains the dependencies necessary for running the project.
