import csv
import os
import pandas as pd
import json

def load_player_data(hitters_path: str, pitchers_path: str):
    """Load hitter and pitcher data into dictionaries keyed by MLBAMID."""
    print("Loading player data...")
    
    # Load hitters
    hitters_dict = {}
    hitters_list = []
    with open(hitters_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mlbamid = row.get('MLBAMID')
            if mlbamid:
                hitters_dict[mlbamid] = row
                hitters_list.append(row)
    
    # Load pitchers
    pitchers_dict = {}
    pitchers_list = []
    with open(pitchers_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mlbamid = row.get('MLBAMID')
            if mlbamid:
                pitchers_dict[mlbamid] = row
                pitchers_list.append(row)
    
    print(f"✅ Loaded {len(hitters_dict)} hitters and {len(pitchers_dict)} pitchers")
    return hitters_dict, pitchers_dict, hitters_list, pitchers_list

def find_player(players, name):
    """Find a player by name (case-insensitive)."""
    for player in players:
        if player['Name'].lower() == name.lower():
            return player
    return None

def get_player_input(players, player_type):
    """Prompt for a player name and find them in the list."""
    while True:
        player_name = input(f"Enter the {player_type}'s name: ")
        player = find_player(players, player_name)
        
        if not player:
            print(f"❌ {player_type.capitalize()} not found in database.")
            retry = input("Try again? (y/n): ").lower()
            if retry != 'y':
                return None
        else:
            print(f"✅ {player_type.capitalize()} found: {player['Name']}")
            return player

def save_matchup_info(pitcher, hitter):
    """Save the pitcher-hitter matchup information for later use."""
    matchup_data = {
        'pitcher': {
            'name': pitcher.get('Name'),
            'mlbamid': pitcher.get('MLBAMID'),
            'team': pitcher.get('Team', 'N/A')
        },
        'hitter': {
            'name': hitter.get('Name'),
            'mlbamid': hitter.get('MLBAMID'),
            'team': hitter.get('Team', 'N/A')
        }
    }
    
    # Save to JSON file
    with open('current_matchup.json', 'w') as f:
        json.dump(matchup_data, f, indent=2)
    
    print(f"\n✅ Matchup information saved to current_matchup.json")
    return matchup_data

def display_matchup_summary(matchup_data):
    """Display a summary of the matchup."""
    print("\n" + "=" * 70)
    print("SELECTED MATCHUP")
    print("=" * 70)
    print(f"\n🥎 Pitcher: {matchup_data['pitcher']['name']}")
    print(f"   Team: {matchup_data['pitcher']['team']}")
    print(f"   MLBAMID: {matchup_data['pitcher']['mlbamid']}")
    print(f"\n⚾ Hitter: {matchup_data['hitter']['name']}")
    print(f"   Team: {matchup_data['hitter']['team']}")
    print(f"   MLBAMID: {matchup_data['hitter']['mlbamid']}")
    print("=" * 70)

def load_encounters(encounters_path: str):
    """Load all encounters from the CSV file."""
    print(f"\nLoading encounters from {encounters_path}...")
    
    encounters = []
    with open(encounters_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            encounters.append(row)
    
    print(f"✅ Loaded {len(encounters)} encounters")
    return encounters

def find_pitcher_stats(pitcher_id, pitchers_dict):
    """Find pitcher statistics by MLBAMID."""
    return pitchers_dict.get(pitcher_id)

def find_hitter_stats(hitter_id, hitters_dict):
    """Find hitter statistics by MLBAMID."""
    return hitters_dict.get(hitter_id)

def combine_encounter_with_player_stats(encounter, pitcher_stats, hitter_stats):
    """
    Combine an encounter with pitcher and hitter statistics.
    Returns a dictionary with prefixed keys.
    """
    combined = {}
    
    # Add pitcher stats with prefix (if available)
    if pitcher_stats:
        for key, value in pitcher_stats.items():
            combined[f"Pitcher_{key}"] = value
    else:
        combined["Pitcher_MLBAMID"] = encounter.get('pitcher', 'unknown')
        combined["Pitcher_Available"] = "No"
    
    # Add hitter stats with prefix (if available)
    if hitter_stats:
        for key, value in hitter_stats.items():
            combined[f"Hitter_{key}"] = value
    else:
        combined["Hitter_MLBAMID"] = encounter.get('batter', 'unknown')
        combined["Hitter_Available"] = "No"
    
    # Add encounter data with prefix
    for key, value in encounter.items():
        combined[f"Encounter_{key}"] = value
    
    return combined


def create_training_dataset(encounters, hitters_dict, pitchers_dict):
    """
    Process all encounters and create training dataset.
    Each encounter becomes one row with pitcher stats + hitter stats + encounter data.
    """
    print("\nProcessing encounters and matching with player statistics...")
    
    training_data = []
    encounters_with_stats = 0
    encounters_missing_pitcher = 0
    encounters_missing_hitter = 0
    encounters_missing_both = 0
    
    for i, encounter in enumerate(encounters):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(encounters)} encounters...")
        
        pitcher_id = encounter.get('pitcher')
        hitter_id = encounter.get('batter')
        
        # Find player statistics
        pitcher_stats = find_pitcher_stats(pitcher_id, pitchers_dict)
        hitter_stats = find_hitter_stats(hitter_id, hitters_dict)
        
        # Track statistics
        has_pitcher = pitcher_stats is not None
        has_hitter = hitter_stats is not None
        
        if has_pitcher and has_hitter:
            encounters_with_stats += 1
        elif not has_pitcher and not has_hitter:
            encounters_missing_both += 1
        elif not has_pitcher:
            encounters_missing_pitcher += 1
        else:
            encounters_missing_hitter += 1
        
        # Combine data (even if some stats are missing)
        combined_row = combine_encounter_with_player_stats(
            encounter, pitcher_stats, hitter_stats
        )
        training_data.append(combined_row)
    
    print(f"\n✅ Processed all {len(encounters)} encounters")
    print(f"\nStatistics:")
    print(f"  ✅ Complete (both pitcher & hitter stats): {encounters_with_stats}")
    print(f"  ⚠️  Missing pitcher stats only: {encounters_missing_pitcher}")
    print(f"  ⚠️  Missing hitter stats only: {encounters_missing_hitter}")
    print(f"  ❌ Missing both stats: {encounters_missing_both}")
    
    return training_data, {
        'total': len(encounters),
        'complete': encounters_with_stats,
        'missing_pitcher': encounters_missing_pitcher,
        'missing_hitter': encounters_missing_hitter,
        'missing_both': encounters_missing_both
    }

def save_training_data(training_data, pitcher_mlbamid, hitter_mlbamid):
    """Save the combined training data to a CSV file with the expected naming format."""
    if not training_data:
        print("❌ No training data to save!")
        return
    
    # Create filename in the format main.py expects
    output_path = f'combined_P{pitcher_mlbamid}_H{hitter_mlbamid}.csv'
    
    print(f"\nSaving training data to {output_path}...")
    
    # Convert to DataFrame for easier saving
    df = pd.DataFrame(training_data)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {len(training_data)} rows to {output_path}")
    print(f"   Columns: {len(df.columns)}")
    
    # Also save as combined_training_data.csv for convenience
    df.to_csv('combined_training_data.csv', index=False)
    print(f"✅ Also saved as combined_training_data.csv")
    
    return df

def main():
    print("=" * 70)
    print("Baseball Training Data Generator - Encounter-First Approach")
    print("=" * 70)
    
    # File paths
    HITTERS_FILE = 'FHH.csv'
    PITCHERS_FILE = 'FHP.csv'
    ENCOUNTERS_FILE = 'previous_encounters.csv'
    
    # Check if files exist
    for filepath in [HITTERS_FILE, PITCHERS_FILE, ENCOUNTERS_FILE]:
        if not os.path.exists(filepath):
            print(f"❌ Error: {filepath} not found!")
            print("Please ensure all required files are in the current directory.")
            return
    
    # Step 0: Get player selection
    print("\n" + "=" * 70)
    print("STEP 0: SELECT PITCHER AND HITTER")
    print("=" * 70)
    print("First, let's verify the pitcher and hitter you want to analyze.")
    print("This information will be saved for later prediction steps.\n")
    
    # Load player data (need lists for selection)
    hitters_dict, pitchers_dict, hitters_list, pitchers_list = load_player_data(HITTERS_FILE, PITCHERS_FILE)
    
    # Get pitcher
    print("\n" + "-" * 70)
    print("PITCHER SELECTION")
    print("-" * 70)
    pitcher = get_player_input(pitchers_list, "pitcher")
    
    if not pitcher:
        print("\n❌ Pitcher selection cancelled. Exiting.")
        return
    
    # Get hitter
    print("\n" + "-" * 70)
    print("HITTER SELECTION")
    print("-" * 70)
    hitter = get_player_input(hitters_list, "hitter")
    
    if not hitter:
        print("\n❌ Hitter selection cancelled. Exiting.")
        return
    
    # Get MLBAMIDs
    pitcher_mlbamid = pitcher.get('MLBAMID')
    hitter_mlbamid = hitter.get('MLBAMID')
    
    # Save matchup information
    matchup_data = save_matchup_info(pitcher, hitter)
    display_matchup_summary(matchup_data)
    
    print("\n💡 Now processing ALL encounters to build training dataset...")
    print("   (Your selected matchup is saved for prediction later)")
    
    # Step 1: Load encounters
    print("\n" + "=" * 70)
    print("STEP 1: LOADING ENCOUNTERS")
    print("=" * 70)
    encounters = load_encounters(ENCOUNTERS_FILE)
    
    # Step 2: Create training dataset
    print("\n" + "=" * 70)
    print("STEP 2: CREATING TRAINING DATASET")
    print("=" * 70)
    training_data, stats = create_training_dataset(encounters, hitters_dict, pitchers_dict)
    
    # Step 3: Save training data
    print("\n" + "=" * 70)
    print("STEP 3: SAVING TRAINING DATA")
    print("=" * 70)
    df = save_training_data(training_data, pitcher_mlbamid, hitter_mlbamid)
    
    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Dataset Summary:")
    print(f"   • Total encounters processed: {stats['total']}")
    print(f"   • Encounters with complete stats: {stats['complete']} ({stats['complete']/stats['total']*100:.1f}%)")
    print(f"   • Missing pitcher stats: {stats['missing_pitcher']} ({stats['missing_pitcher']/stats['total']*100:.1f}%)")
    print(f"   • Missing hitter stats: {stats['missing_hitter']} ({stats['missing_hitter']/stats['total']*100:.1f}%)")
    print(f"   • Missing both stats: {stats['missing_both']} ({stats['missing_both']/stats['total']*100:.1f}%)")
    
    print(f"\n📁 Output files:")
    print(f"   • combined_P{pitcher_mlbamid}_H{hitter_mlbamid}.csv")
    print(f"   • combined_training_data.csv")
    print(f"   • current_matchup.json")
    print(f"   Total rows: {len(df)} | Columns: {len(df.columns)}")
    
    print(f"\n📁 Matchup saved:")
    print(f"   • Pitcher: {matchup_data['pitcher']['name']}")
    print(f"   • Hitter: {matchup_data['hitter']['name']}")
    
    print("\n✅ Next step: Pipeline will automatically run prepare_dataset.py")
   
if __name__ == "__main__":
    main()
