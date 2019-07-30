import os
import pytest
from network_wrangler import ProjectCard
from network_wrangler import Scenario
from network_wrangler.Logger import WranglerLogger

"""
Run just the tests labeled scenario using `pytest -v -m scenario`
To run with print statments, use `pytest -s -m scenario`
"""

@pytest.mark.scenario
def test_project_card_read():
    print("test_project_card_read: Testing project card is read into object dictionary")
    in_dir  = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'example', 'stpaul','project_cards')
    in_file = os.path.join(in_dir,"1_simple_roadway_attribute_change.yml")
    project_card = ProjectCard.read(in_file)
    WranglerLogger.info(project_card.__dict__)
    print("---test_project_card_read()---\n",str(project_card),"\n---end test_project_card_read()---\n")
    assert(project_card.category == "Roadway Attribute Change")

@pytest.mark.ashish
@pytest.mark.scenario
def test_scenario_conflicts():
    base_scenario = {}

    project_cards_list = []
    project_cards_list.append(ProjectCard.read(os.path.join(os.getcwd(),'example','stpaul','project_cards','4_test_project_card.yml')))
    project_cards_list.append(ProjectCard.read(os.path.join(os.getcwd(),'example','stpaul','project_cards','5_test_project_card.yml')))
    project_cards_list.append(ProjectCard.read(os.path.join(os.getcwd(),'example','stpaul','project_cards','6_test_project_card.yml')))

    scen = Scenario.create_scenario(base_scenario = base_scenario, project_cards_list = project_cards_list)
    scen.check_scenario_conflicts()

@pytest.mark.ashish
@pytest.mark.scenario
def test_scenario_requisites():
    base_scenario = {}

    project_cards_list = []
    project_cards_list.append(ProjectCard.read(os.path.join(os.getcwd(),'example','stpaul','project_cards','4_test_project_card.yml')))
    project_cards_list.append(ProjectCard.read(os.path.join(os.getcwd(),'example','stpaul','project_cards','5_test_project_card.yml')))
    project_cards_list.append(ProjectCard.read(os.path.join(os.getcwd(),'example','stpaul','project_cards','6_test_project_card.yml')))

    scen = Scenario.create_scenario(base_scenario = base_scenario, project_cards_list = project_cards_list)
    scen.check_scenario_requisites()

@pytest.mark.topo
@pytest.mark.scenario
def test_project_sort():
    base_scenario = {}

    project_cards_list = []
    project_cards_list.append(ProjectCard.read(os.path.join(os.getcwd(),'example','stpaul','project_cards','4_test_project_card.yml')))
    project_cards_list.append(ProjectCard.read(os.path.join(os.getcwd(),'example','stpaul','project_cards','5_test_project_card.yml')))
    project_cards_list.append(ProjectCard.read(os.path.join(os.getcwd(),'example','stpaul','project_cards','6_test_project_card.yml')))

    scen = Scenario.create_scenario(base_scenario = base_scenario, project_cards_list = project_cards_list)

    scen.check_scenario_conflicts()
    scen.check_scenario_requisites()

    sorted_projects_cards = scen.create_ordered_project_cards()
    print([project_card.name for project_card in sorted_projects_cards])
