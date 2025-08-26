/*
 * simulation.cpp
 *
 *  Created on: 28-aug-2017
 *      Author: M. El-Kebir
 */

#include <iostream>
#include <fstream>
#include "simulation.h"
#include "beta_distribution.hpp"
#include <lemon/bfs.h>

// constructor
Simulation::Simulation(double K,
                       double migrationRate,
                       double mutationRate,
                       double driverProb,
                       double mutFreqThreshold,
                       int maxNrAnatomicalSites,
                       int nrSamplesPerAnatomicalSite,
                       int nrSamplesPrimary,
                       int targetCoverage,
                       Pattern pattern,
                       double seqErrorRate,
                       double purity)
  // initialization
  : _G() 
  , _rootG(lemon::INVALID)
  , _anatomicalSiteMap(_G)
  , _indexToVertexG()
  , _seedingCells()
  , _arcToCell(_G)
  , _generation(0)
  , _extantCellsByDrivers()
  , _nrExtantCells(0)
  , _nrMutations(0)
  , _nrAnatomicalSites(0)
  , _nrActiveAnatomicalSites(0)
  , _isActiveAnatomicalSite(0)
  , _anatomicalSiteFactors()
  , _K(K)
  , _migrationRate(migrationRate)
  , _mutationRate(mutationRate)
  , _driverProb(driverProb)
  , _mutFreqThreshold(mutFreqThreshold)
  , _maxNrAnatomicalSites(maxNrAnatomicalSites)
  , _nrSamplesPerAnatomicalSite(nrSamplesPerAnatomicalSite)
  , _nrSamplesPrimary(nrSamplesPrimary)
  , _targetCoverage(targetCoverage)
  , _pattern(pattern)
  , _seqErrorRate(seqErrorRate)
  , _purity(purity)
  , _populationRecord()
  , _driverMutations()
  , _observableMutationToCellTreeMutation()
  , _cellTreeMutationToObservableMutation()
  , _pCloneT(NULL)
  , _pAnatomicalSiteProportions(NULL)
  , _pSampleProportions(NULL)
  , _pMutations(NULL)
  , _pAnatomicalSite(NULL)
  , _freq()
  , _var()
  , _ref()
  , _sampledLeaves()
  , _pCloneTT(NULL)
  , _pSampleProportionsTT(NULL)
  , _pMutationsTT(NULL)
  , _pAnatomicalSiteTT(NULL)
  , _sampledMutations()
  , _sampledMutationsByAnatomicalSite()
  , _sampledDriverMutations()
  , _pVertexLabelingTT(NULL)
  , _arcGtoVerticesTT(_G)
  , _arcGtoSampledMutations(_G)
  //, _cloneCells() // *
{
  init();
}

Simulation::~Simulation() // destructor
{
  delete _pAnatomicalSiteProportions;
  delete _pMutations;
  delete _pAnatomicalSite;
  delete _pCloneT;
  
  delete _pSampleProportionsTT;
  delete _pMutationsTT;
  delete _pAnatomicalSiteTT;
  delete _pVertexLabelingTT;
  delete _pCloneTT;
}

// simulate reads
bool Simulation::simulateReadCounts()
{
  typedef std::vector<NodeSet> NodeSetVector; // define NodeSetVector as a vector of nodes
  typedef std::map<int, NodeSet> NodeSetMap;  // define NodeSetMap as {#:nodes}

  /// 1. Partition leaf set into anatomical sites
  NodeSetVector leavesPerAnatomicalSite(_nrAnatomicalSites);  // init leavesPerAnatomicalSite
  NodeSetMap leavesPerMutation; // init leavesPerMutation = {mut#:nodes}
  for (Node v : _pCloneT->leafSet())  // loop over leaf set of pCloneT
  {
    int s = (*_pAnatomicalSite)[v]; // get site s for node v
    if (s == 0) // primary site
    {
      (*_pSampleProportions)[v] = DoubleVector(_nrSamplesPrimary, 0); // init sample proportions for primary site as 0
    }
    else
    {
      (*_pSampleProportions)[v] = DoubleVector(_nrSamplesPerAnatomicalSite, 0); // init sample proportions for other sites as 0
    }
    
    if ((*_pAnatomicalSiteProportions)[v] > _mutFreqThreshold)  // proportion of node v at site s > mutFreqThreshold
    {
      leavesPerAnatomicalSite[s].insert(v); // 1. add node v to site s
      for (int j : (*_pMutations)[v]) // loop over mutations of node v
      {
        if (leavesPerMutation.count(j) == 0)  // mut j not present
        {
          leavesPerMutation[j] = NodeSet(); // init j
        }
        leavesPerMutation[j].insert(v); // add node v to mut j
      }
    }
  }
  
  const int nrMutations = leavesPerMutation.size(); // # of muts
  
  std::poisson_distribution<> poisson(_targetCoverage); // poisson dist with u = targetCov
  
  _freq = DoubleTensor(_nrAnatomicalSites,
                       DoubleMatrix(_nrSamplesPerAnatomicalSite,
                                    DoubleVector(nrMutations, 0))); // init freq as sites x samples x muts
  _freq[0] = DoubleMatrix(_nrSamplesPrimary, DoubleVector(nrMutations, 0)); // init freq at primary site
  
  _var = IntTensor(_nrAnatomicalSites,
                   IntMatrix(_nrSamplesPerAnatomicalSite,
                             IntVector(nrMutations, 0))); // init var as sites x samples x muts
  _var[0] = IntMatrix(_nrSamplesPrimary,
                      IntVector(nrMutations, 0)); // init var at primary site
  
  _ref = IntTensor(_nrAnatomicalSites,
                   IntMatrix(_nrSamplesPerAnatomicalSite,
                             IntVector(nrMutations, 0))); // init ref as sites x samples x muts
  _ref[0] = IntMatrix(_nrSamplesPrimary,
                      IntVector(nrMutations, 0)); // init ref at primary site
  
  _sampledLeaves.clear();
  
  int pp = 0;
  for (int s = 0; s < _nrAnatomicalSites; ++s)  // loop over site s
  {
    if (!_isActiveAnatomicalSite[s]) continue;  // check if s is active
    const int k_s = s == 0 ? _nrSamplesPrimary : _nrSamplesPerAnatomicalSite; // get # of samples based on primary or other sites 
    for (int p = 0; p < k_s; ++p, ++pp) // loop over samples p at site s
    {
      // simulate draw from Dirichlet by drawing from gamma distributions
      // https://en.wikipedia.org/wiki/Dirichlet_distribution#Gamma_distribution
      
      DoubleNodeMap draw(_pCloneT->tree(), 0);  // init draw as {nodes:0}
      double sum = 0;
      for (Node v : leavesPerAnatomicalSite[s]) // loop over nodes v at site s
      {
        std::gamma_distribution<> gamma_dist((*_pAnatomicalSiteProportions)[v]*3, 1); // gamma dist
        draw[v] = (gamma_dist(g_rng));  // draw a random value for node v
        sum += draw[v]; // add to sum
      }
      
      double new_sum = 0;
      for (Node v : leavesPerAnatomicalSite[s]) // loop over nodes v at site s
      {
        if ((draw[v] / sum) >= 0.1) // draw for node v > 0.1
        {
          (*_pSampleProportions)[v][p] = draw[v]; // set sample proportion for node v at sample p
          new_sum += draw[v]; // add to new sum
          _sampledLeaves.insert(v); // add node v to sampled leaves
        }
        else
        {
          (*_pSampleProportions)[v][p] = 0; // sample proportion = 0
        }
      }
      
//      std::cout << "s = " << s << " p = " << p << std::endl;
      for (Node v : leavesPerAnatomicalSite[s]) // loop over nodes v at site s
      {
        (*_pSampleProportions)[v][p] /= new_sum;  // normalize by new sum
//        std::cout << (*_pSampleProportions)[v][p] << std::endl;
//        assert(0 <= (*_pSampleProportions)[v][p]);
//        assert((*_pSampleProportions)[v][p] <= 1);
      }
      
      for (int i = 0; i < nrMutations; ++i) // loop over muts i
      {
        for (Node v : leavesPerMutation[i]) // loop over nodes v with mut i
        {
          if ((*_pAnatomicalSite)[v] == s)  // node v is at site s
          {
            _freq[s][p][i] += (*_pSampleProportions)[v][p]; // set freq for site s, sample p, mut i
          }
//          assert(_freq[s][p][i] <= 1);
        }
        
        int coverage = poisson(g_rng);  // random from poisson dist
        
        // if freq < 0.1 then zero out
        if (_freq[s][p][i] < 0.1)
        {
          _freq[s][p][i] = 0;
        }
        
        // divide 2, heterozygous diploid
        double f = (_freq[s][p][i] * _purity) / 2;
        
//        int alpha = std::max(1, int(round(f * 100)));
//        int beta = 100 - alpha;
//        sftrabbit::beta_distribution<> beta_dist(alpha, beta);
//        double ff = f == 0 ? 0 : beta_dist(g_rng);
        
        std::binomial_distribution<> binom(coverage, f);  // coverage = # of trials; f = prob 
//        std::binomial_distribution<> binom(coverage, ff);

        int org_var = binom(g_rng); // original variant from binomial dist
        int org_ref = coverage - _var[s][p][i]; // reference + variant = coverage

        if (g_tol.nonZero(_seqErrorRate)) // sequencing errors
        {
          std::binomial_distribution<> binom_noise_var(org_var,
                                                       _seqErrorRate);
          std::binomial_distribution<> binom_noise_ref(org_ref,
                                                       _seqErrorRate);

          int flips_var =  binom_noise_var(g_rng);  // # of errors in variants
          int flips_ref =  binom_noise_ref(g_rng);  // # of errors in references
          
          _var[s][p][i] = org_var - flips_var + flips_ref;
          _ref[s][p][i] = coverage - _var[s][p][i];
        }
        else  // no errors
        {
          _var[s][p][i] = org_var;
          _ref[s][p][i] = org_ref;
        }
      }
    }
  }
  
  return constructSampledCloneTree();
}

void Simulation::writeDrivers(std::ostream& out) const
{
  for (int j : _sampledMutations)
  {
    int i = _observableMutationToCellTreeMutation[j];
    if (_driverMutations.count(i) == 1)
    {
      out << j << std::endl;
    }
  }
  
  out << "Number of generations: " << _generation << std::endl;
  out << "Number of mutations: " << _nrMutations << std::endl;
  out << "Number of extant cells: " << _nrExtantCells << std::endl;
  out << "Number of active anatomical sites: " << _nrActiveAnatomicalSites << std::endl;
}

void Simulation::writeReadCounts(std::ostream& out) const
{
  int nrMutations = _sampledMutations.size();

  if (_nrSamplesPerAnatomicalSite == 0)
  {
    out << " 1 #anatomical sites" << std::endl;
  }
  else
  {
    out << _nrActiveAnatomicalSites << " #anatomical sites" << std::endl;
  }
  out << _nrSamplesPrimary + _nrSamplesPerAnatomicalSite * (_nrActiveAnatomicalSites - 1)
      << " #samples" << std::endl;
  out << nrMutations << " #mutations" << std::endl;
  
  out << "#sample_index\tsample_label\tanatomical_site_index\tanatomical_site_label"\
  "\tcharacter_index\tcharacter_label\tref\tvar" << std::endl;

  char buf[1024];
  int activeSiteIndex = 0;
  int pp = 0;
  for (int s = 0; s < _nrAnatomicalSites; ++s)
  {
    if (!_isActiveAnatomicalSite[s])
      continue;
    
    std::string label_s = _anatomicalSiteLabel[s];
    
    const int k_s = s == 0 ? _nrSamplesPrimary : _nrSamplesPerAnatomicalSite;
    for (int p = 0; p < k_s; ++p, ++pp)
    {
      snprintf(buf, 1024, "%s_%d", label_s.c_str(), p);
      std::string label_p = buf;
      int mutationIdx = 0;
      for (int i : _sampledMutations)
      {
        out << pp << "\t" << label_p << "\t"
            << activeSiteIndex << "\t" << label_s << "\t"
            << mutationIdx << "\t" << i << "\t"
            << _ref[s][p][i] << "\t" << _var[s][p][i] << std::endl;
        ++mutationIdx;
      }
    }
    ++activeSiteIndex;
  }
}

void Simulation::init()
{
  IntVector initialPassengerMutations;
  int initialMutation = 0;
  _driverMutations.clear();
  _driverMutations.insert(initialMutation);
  _nrMutations = 1;
  
  int initialAnatomicalSite = 0;
  _isActiveAnatomicalSite.push_back(true);
  _nrAnatomicalSites = 1;
  _nrActiveAnatomicalSites = 1;
  
  Cell founder = Cell(initialPassengerMutations,
                      initialMutation,
                      initialAnatomicalSite,
                      0,  // *id
                      0,  // *generation
                      -1);  // *parent
  
  _extantCellsByDrivers.push_back(std::map<IntSet, CellVector>());  // initialize a new vector with driver muts and cells
  _extantCellsByDrivers[0][_driverMutations].push_back(founder);  // add init driver muts and founder cells
  
  _G.clear();
  _rootG = _G.addNode();
  _anatomicalSiteMap[_rootG] = 0;
  _indexToVertexG.clear();
  _indexToVertexG.push_back(_rootG);
  
  _generation = 0;
  _nrExtantCells = 1;
  
  _seedingCells.clear();
  _seedingCells.push_back(CellVector(1, founder));
}

// get (1 - N / K) for each clone; count cells at each site.
void Simulation::updateAnatomicalSiteFactors()
{
  // this is calculating 1 - N / K for each site
  // and updates the population record
  
  _anatomicalSiteFactors = AnatomicalSiteFactorVector(_nrAnatomicalSites, AnatomicalSiteMap());
  _nrActiveAnatomicalSites = 0;
  for (int s = 0; s < _nrAnatomicalSites; ++s)  // loop over sites
  {
    _isActiveAnatomicalSite[s] = false;
    int nrCells_s = 0;
    
    double sum = 0;
    IntSetVector toRemove;
    for (const auto& kv : _extantCellsByDrivers[s]) // loop over cells at s
    {
      if (kv.second.empty())  // no cells
      {
        toRemove.push_back(kv.first);
      }
      else
      {
        sum += kv.first.size(); // # of muts
      }
    }
    
    for (const IntSet& drivers : toRemove)
    {
      _extantCellsByDrivers[s].erase(drivers);  // remove muts w/o cells
    }
    
    for (const auto& kv : _extantCellsByDrivers[s]) // loop over cells
    {
      const IntSet& X = kv.first; // muts X
      int nrCells_sX = kv.second.size();  // # of cells N
      nrCells_s += nrCells_sX;  // total # of cells
      
      _anatomicalSiteFactors[s][X] = std::max(0.0,
                                              1 - double(nrCells_sX)
                                                / (_K * X.size())); // (1 - N / K) for muts X at site s
    }
    
    if (nrCells_s > _K / 10)  // total # > K / 10
    {
      ++_nrActiveAnatomicalSites;
      _isActiveAnatomicalSite[s] = true;  // set as active site
    }

    if (s < _populationRecord.size())
    {
      _populationRecord[s].push_back(nrCells_s);  // add # to record
    }
    else
    {
      _populationRecord.push_back(IntVector(_generation, 0)); // add new record
      _populationRecord[s].push_back(nrCells_s);
    }
  }
}

// get migration target site
int Simulation::getTargetAnatomicalSite(int s)
{
  lemon::DynArcLookUp<Digraph> arcLookUp(_G);
  std::uniform_real_distribution<> unif(0, 1);
  
  Node v_s = _indexToVertexG[s];
  
  int t = -1;
  switch (_pattern)
  {
    case PATTERN_mS:
      t = _nrAnatomicalSites;
      break;
    case PATTERN_S:
      if (OutArcIt(_G, v_s) == lemon::INVALID || unif(g_rng) < 0.5)
      {
        t = _nrAnatomicalSites;
      }
      else
      {
        IntVector potentialTargets;
        for (OutArcIt a(_G, v_s); a != lemon::INVALID; ++a)
        {
          Node v_t = _G.target(a);
          potentialTargets.push_back(_anatomicalSiteMap[v_t]);
        }
        
        std::uniform_int_distribution<> unif_int(0, potentialTargets.size() - 1);
        
        t = potentialTargets[unif_int(g_rng)];
      }
      break;
    case PATTERN_M:
      {
        IntVector potentialTargets;
        for (NodeIt v_t(_G); v_t != lemon::INVALID; ++v_t)
        {
          if (v_t == v_s) continue;
          Arc tmp = _G.addArc(v_s, v_t);
          if (lemon::dag(_G))
          {
            potentialTargets.push_back(_anatomicalSiteMap[v_t]);
            if (arcLookUp(v_s, v_t) == lemon::INVALID)
            {
              // make it more likely to have multi-source seeding
              potentialTargets.push_back(_anatomicalSiteMap[v_t]);
              potentialTargets.push_back(_anatomicalSiteMap[v_t]);
              potentialTargets.push_back(_anatomicalSiteMap[v_t]);
              potentialTargets.push_back(_anatomicalSiteMap[v_t]);
            }
          }
          _G.erase(tmp);
        }
        
        if (potentialTargets.empty() || unif(g_rng) < 0.5)
        {
          t = _nrAnatomicalSites;
        }
        else
        {
          std::uniform_int_distribution<> unif_int(0, potentialTargets.size() - 1);
          
          t = potentialTargets[unif_int(g_rng)];
        }
      }
      break;
    case PATTERN_R:
      {
        IntVector potentialTargets;
        for (NodeIt v_t(_G); v_t != lemon::INVALID; ++v_t)
        {
          if (v_t == v_s) continue;
          potentialTargets.push_back(_anatomicalSiteMap[v_t]);
          Arc tmp = _G.addArc(v_s, v_t);
          if (!lemon::dag(_G))
          {
            // make it more likely to have reseeding
            potentialTargets.push_back(_anatomicalSiteMap[v_t]);
            potentialTargets.push_back(_anatomicalSiteMap[v_t]);
            potentialTargets.push_back(_anatomicalSiteMap[v_t]);
            potentialTargets.push_back(_anatomicalSiteMap[v_t]);
          }
          else if (arcLookUp(v_s, v_t) == lemon::INVALID)
          {
            potentialTargets.push_back(_anatomicalSiteMap[v_t]);
          }
          _G.erase(tmp);
        }

        if (potentialTargets.empty() || unif(g_rng) < 0.5)
        {
          t = _nrAnatomicalSites;
        }
        else
        {
          std::uniform_int_distribution<> unif_int(0, potentialTargets.size() - 1);
          
          t = potentialTargets[unif_int(g_rng)];
        }
      }
      break;
    default:
      return 0;
  }
  
  if (t == _nrAnatomicalSites)
  {
    Node v_t = _G.addNode();
    _indexToVertexG.push_back(v_t);
    _anatomicalSiteMap[v_t] = t;
    _extantCellsByDrivers.push_back(std::map<IntSet, CellVector>());
    _seedingCells.push_back(CellVector());
    _isActiveAnatomicalSite.push_back(false);
    ++_nrAnatomicalSites;
  }

  return t;
}

// migration
void Simulation::migrate()
{
  std::uniform_real_distribution<> unif(0, 1);
  std::poisson_distribution<> poisson(1);
  int currentNrAnatomicalSites = _nrAnatomicalSites;
  
  for (int s = 0; s < currentNrAnatomicalSites; ++s)
  {
    Node v_s = _indexToVertexG[s];
    int nrExtantCells_s = getNrExtantCells(s);

    // no migration with probability 1 - nrExtantCells_s * _migrationRate
    if (unif(g_rng) > getNrWeightedExtantCells(s) * _migrationRate)
      continue;
    
    int migrationCount = 0;
    switch (_pattern)
    {
      case PATTERN_mS:
        migrationCount = 1;
        break;
      default:
        migrationCount = 1 + poisson(g_rng);
    }
    
    if (migrationCount > nrExtantCells_s)
      migrationCount = nrExtantCells_s;
    
    // last migrationCount cells will migrate
    int t = getTargetAnatomicalSite(s);
    Node v_t = _indexToVertexG[t];

    IntVector migrationVector = draw(s, migrationCount);  // draw migrating cells for each clone
    int idx = 0;
    for (auto& kv : _extantCellsByDrivers[s]) // loop over each clone
    {
      const IntSet& X = kv.first;
      CellVector& extantCellByDrivers_sX = kv.second;
      if (migrationVector[idx] > 0)
      {
        std::shuffle(extantCellByDrivers_sX.begin(),
                     extantCellByDrivers_sX.end(),
                     g_rng);
      }
      IntSet migratedClones;
      for (int i = 0; i < migrationVector[idx]; ++i)  // migrate cells for this clone
      {
        Cell& cell = extantCellByDrivers_sX.back();
        if (migratedClones.count(cell.getMutation()) == 1)
        {
          continue;
        }
        migratedClones.insert(cell.getMutation());
        cell.migrate(t);
        Arc a = _G.addArc(v_s, v_t);
        _arcToCell[a] = std::make_pair(cell, X);
        _seedingCells[t].push_back(cell);
        _extantCellsByDrivers[t][X].push_back(cell);
        extantCellByDrivers_sX.pop_back();
      }
      ++idx;
    }
  }
}

// birth with muts
bool Simulation::simulate(bool verbose)
{
  std::uniform_real_distribution<> unif(0, 1);
  
  init();
  int id = 0;  // *init id
  CellVector cells; // *all cells

  while (_nrExtantCells > 0)
  {
    if (_nrActiveAnatomicalSites > _maxNrAnatomicalSites)
    {
      if (verbose)
      {
        std::cout << "Maximum number of active anatomical sites reached" << std::endl;
        
        // *output cell division history
        std::ofstream division_outfile("cellDivisionHistory.txt");
        for (const Cell& cell: cells)
        {
          //std::cout << cell.getGeneration() << "\t" << cell.getParent() << "\t" << cell.getID() << std::endl;
          division_outfile << cell.getGeneration() << "\t" << cell.getParent() << "\t" << cell.getID() << std::endl;
        }
        division_outfile.close();  
        
        // *output migration cells
        std::ofstream migration_outfile("migrationCells.txt");
        int s = 0;
        for (const CellVector& migratingCells: _seedingCells)
        {
          //std::cout << "Site " << s << ": ";
          migration_outfile << "Site " << s << ": ";
          for (const Cell& cell: migratingCells)
          {
            //std::cout << cell.getID() << ",";
            migration_outfile << cell.getID() << ",";
          }
          //std::cout << std::endl;
          migration_outfile << std::endl;
          ++s;
        }
        migration_outfile.close();
      }
      break;
    }
    
    if (verbose)// && (_generation % 20 == 0 || _nrExtantCells > 500))
    {
      std::cout << "Generation: " << _generation
        << ", #cells: " << _nrExtantCells
        << ", #anatomical sites: " << _nrActiveAnatomicalSites << "/" << _nrAnatomicalSites
        << " (" << _driverMutations.size() << "/" << _nrMutations << ")"
        << std::endl;
      
      for (int s = 0; s < _nrAnatomicalSites; ++s)
      {
        if (!_isActiveAnatomicalSite[s]) continue;
        for (const auto& kv : _extantCellsByDrivers[s])
        {
          if (kv.second.size() > 1000)
          {
            const IntSet& driverMutations = kv.first;
            std::cout << "Anatomical site: " << s << " ";
            std::cout << "{";
            for (int i : driverMutations)
            {
              std::cout << " " << i;
            }
            std::cout << " }, number of cells: "
                      << kv.second.size()
                      << std::endl;
          }
        }
      }
    }
    
    // update site factors (k - N) / k and update site pop record
    updateAnatomicalSiteFactors();
    
    // divide and mutate
    ClonalComposition newExtantCellsByDrivers;

    int newNrExtantCells = 0;
    for (int s = 0; s < _nrAnatomicalSites; ++s)  // loop over site
    {
      newExtantCellsByDrivers.push_back(std::map<IntSet, CellVector>());  // new {muts:cells} for s
      std::map<IntSet, CellVector>& newExtantCellsByDrivers_s = newExtantCellsByDrivers.back(); // reference (another name)
      
      for (auto& kv : _extantCellsByDrivers[s]) // loop over cells at s
      {
        const IntSet& X = kv.first; // get mut X
        assert(_anatomicalSiteFactors[s].count(X) == 1);
        double logisticFactor = _anatomicalSiteFactors[s][X]; // factor (1 - N / K)
        
        CellVector& newExtantCellsByDrivers_sX = newExtantCellsByDrivers_s[X];  // reference for cells
        newExtantCellsByDrivers_sX.reserve(_K * X.size());  // reserve place for new cells
        
        for (const Cell& cell : kv.second)  // loop over cells
        {
          switch (cell.performGeneration(logisticFactor, X.size())) // birth or death
          {
            case Cell::REPLICATION:
              {
                ++id; // *update id for new cell 1

                // *create and add daughter cell 1 (same as parent)
                Cell daughterCell1(cell.getPassengerMutations(),
                                   cell.getMutation(),
                                   cell.getAnatomicalSite(),
                                   id,
                                   _generation,
                                   cell.getID());
                
                cells.push_back(daughterCell1);
                newExtantCellsByDrivers_sX.push_back(daughterCell1);
                
                int nrInitialPassengers = floor(_mutationRate); // # of init pass: floor(0.1) = 0
                IntVector initialPassengerMutations;
                for (int i = 0; i < nrInitialPassengers; ++i) // loop over init pass
                {
                  int new_mut = getNewMutation();
                  initialPassengerMutations.push_back(new_mut); // add init pass
                }
                
                ++id; // *update id for new cell 2

                /// do we have a new driver?
                double r = unif(g_rng); // random from uniform dist
                if (r < _driverProb * (X.size() + 1)) // new driver: driverProb = 2 x 10^(-7)
                {
                  IntVector passengerMutations2 = cell.getPassengerMutations(); // get old pass
                  passengerMutations2.insert(passengerMutations2.end(),
                                             initialPassengerMutations.begin(),
                                             initialPassengerMutations.end());  // old pass + init pass (0)

                  int new_mut = getNewMutation(); // get new driver
                  
                  IntSet driverMutations2 = X;  // get old driver
                  driverMutations2.insert(new_mut); // old driver + new driver
                  _driverMutations.insert(new_mut); // all drivers
                  
                  // *create and add cell daughter 2 (with mut)
                  Cell daughterCell2(passengerMutations2,
                                     new_mut,
                                     cell.getAnatomicalSite(),
                                     id,
                                     _generation,
                                     cell.getID());
                  
                  cells.push_back(daughterCell2);
                  newExtantCellsByDrivers_s[driverMutations2].push_back(daughterCell2); // add new cell

                  // *print out new mutation
                  std::cout << "New driver {" << new_mut << "} in " << id << std::endl;
                }
                else if (r < _mutationRate - floor(_mutationRate)) // new pass: 0.1 - 0 = 0.1
                {
                  int new_mut = getNewMutation(); // get new pass
                  
                  IntVector passengerMutations2 = cell.getPassengerMutations(); // get old pass
                  passengerMutations2.insert(passengerMutations2.end(),
                                             initialPassengerMutations.begin(),
                                             initialPassengerMutations.end());  // old pass + init pass (0)
                  
                  passengerMutations2.push_back(new_mut); // add new pass
                  
                  // *create and add cell daughter 2 (with mut)
                  Cell daughterCell2(passengerMutations2,
                                     new_mut,
                                     cell.getAnatomicalSite(),
                                     id,
                                     _generation,
                                     cell.getID());
                  
                  cells.push_back(daughterCell2);
                  newExtantCellsByDrivers_sX.push_back(daughterCell2);  // add new cell

                  // *print out new mutation
                  std::cout << "New passenger {" << new_mut << "} in " << id << std::endl;
                }
                else  // no new mut
                {
                  int new_mut = initialPassengerMutations.empty() ?
                                cell.getMutation() : initialPassengerMutations.back();  // get init pass (-1)
                  
                  IntVector passengerMutations2 = cell.getPassengerMutations(); // get old pass
                  passengerMutations2.insert(passengerMutations2.end(),
                                             initialPassengerMutations.begin(),
                                             initialPassengerMutations.end());  // old pass + init pass (-1)
                  
                  // *create and add cell daughter 2 (with mut)
                  Cell daughterCell2(passengerMutations2,
                                     new_mut,
                                     cell.getAnatomicalSite(),
                                     id,
                                     _generation,
                                     cell.getID());
                  
                  cells.push_back(daughterCell2);
                  newExtantCellsByDrivers_sX.push_back(daughterCell2);  // add new cell
                }
                
                newNrExtantCells += 2;  // 2 daughters
              }
              break;
            case Cell::DEATH:
              break;
          }
        }
      }
    }

    _nrExtantCells = newNrExtantCells;  // new cell #
    
    std::swap(_extantCellsByDrivers, newExtantCellsByDrivers);  // new cells by drivers
    
    migrate();
    
    // new generation
    ++_generation;
  }
  
  updateAnatomicalSiteFactors();
  
  if (_nrExtantCells > 0)
  {
    char buf[1024];
    int idx = 1;
    
    // label sites
    _anatomicalSiteLabel.clear();
    for (int s = 0; s < _nrAnatomicalSites; ++s)
    {
      _anatomicalSiteLabel.push_back("");
      if (_isActiveAnatomicalSite[s])
      {
        if (s == 0)
        {
          _anatomicalSiteLabel.back() = "P";
        }
        else
        {
          snprintf(buf, 1024, "M%d", idx);
          _anatomicalSiteLabel.back() = buf;
          ++idx;
        }
      }
    }
    
    if (!_isActiveAnatomicalSite[0])
    {
      // NO PRIMARY
      std::cerr << "No primary!" << std::endl;
      return false;
    }
    
    constructCloneTree();
    if (!simulateReadCounts())
    {
      return false;
    }
    if (fakeParallelEdges())
    {
      return false;
    }
    else
    {
      return true;
    }
  }
  
  if (verbose)
  {
    std::cout << "No tumor" << std::endl;
  }
  
  return false;
}

// fake parallel edges
bool Simulation::fakeParallelEdges() const
{
  const Digraph& T = _pCloneTT->tree();
  for (NodeIt v_i(T); v_i != lemon::INVALID; ++v_i) // loop over nodes v_i
  {
    if (_pCloneTT->isLeaf(v_i)) continue; // skip leaves
    const std::string& s_i = (*_pVertexLabelingTT)[v_i];  // get v_i label
    
    for (OutArcIt a_ij(T, v_i); a_ij != lemon::INVALID; ++a_ij) // loop over arcs a_ij from source node v_i
    {
      Node v_j = T.target(a_ij);  // get target node v_j
      const std::string& s_j = (*_pVertexLabelingTT)[v_j];  // get v_j label
      for (OutArcIt a_il(T, v_i); a_il != lemon::INVALID; ++a_il) // loop over arcs a_il from node v_i
      {
        if (T.id(a_ij) < T.id(a_il))  // prevent duplication
        {
          Node v_l = T.target(a_il);  // get target node v_l
          const std::string& s_l = (*_pVertexLabelingTT)[v_l];  // get v_l label
          
          if (s_i != s_j && s_j == s_l) // fake parallel edge
          {
            return true;
          }
        }
      }
    }
  }
  return false;
}

MigrationGraph Simulation::getObservedMigrationGraph() const
{ 
  StringNodeMap label(_G);
  for (NodeIt v_s(_G); v_s != lemon::INVALID; ++v_s)
  {
    int s = _anatomicalSiteMap[v_s];
    label[v_s] = _anatomicalSiteLabel[s];
  }
  
  BoolNodeMap filterNodes(_G, false);
  BoolArcMap filterArcs(_G, false);
  SubDigraph subG(_G, filterNodes, filterArcs);
  
  for (NodeIt v_s(_G); v_s != lemon::INVALID; ++v_s)
  {
    int s = _anatomicalSiteMap[v_s];
    filterNodes[v_s] = _isActiveAnatomicalSite[s];
  }
  
  for (ArcIt a(_G); a != lemon::INVALID; ++a)
  {
    if (!_arcGtoVerticesTT[a].empty())
    {
      filterArcs[a] = true;
    }
  }
  
  Node newRoot = _rootG;
  if (!_anatomicalSiteMap[_rootG])
  {
    // find new root
    for (SubNodeIt v(subG); v != lemon::INVALID; ++v)
    {
      if (SubInArcIt(subG, v) == lemon::INVALID)
      {
        newRoot = v;
        break;
      }
    }
  }
  
  Digraph GG;
  Node rootGG;
  StringNodeMap labelGG(GG);
  lemon::digraphCopy(subG, GG)
    .node(newRoot, rootGG)
    .nodeMap(label, labelGG)
    .run();
  
  
  return MigrationGraph(GG, rootGG, labelGG);
}

MigrationGraph Simulation::getMigrationGraph() const
{
  StringNodeMap label(_G);
  for (NodeIt v_s(_G); v_s != lemon::INVALID; ++v_s)
  {
    int s = _anatomicalSiteMap[v_s];
    label[v_s] = _anatomicalSiteLabel[s];
  }

  //for (ArcIt a(_G); a != lemon::INVALID; ++a)
  //{
  //  const Cell& cell = _arcToCell[a];
  //  Node v_s = _G.source(a);
  //  Node v_t = _G.target(a);
  //  int t = _anatomicalSiteMap[v_t];
  //  int i = cell.getMutation();
  //  int observed_i = _cellTreeMutationToObservableMutation.count(i) == 1 ? _cellTreeMutationToObservableMutation.find(i)->second : -1;
  //  std::cout << label[v_s] << " -> " << label[v_t] << " : " << observed_i;
  //  if (_sampledMutations.count(observed_i) == 1)
  //  {
  //    std::cout << ", sampled";
  //  }
  //  if (_sampledMutationsByAnatomicalSite[t].count(observed_i) == 1)
  //  {
  //    std::cout << ", sampled in " << label[v_t];
  //  }

  //  std::cout << std::endl;
  //}
  
  return MigrationGraph(_G, _rootG, label);
}

void Simulation::constructCloneTree()
{
  // 1. Compute mutation frequencies in each anatomical site
  // 2. Infer observable mutations
  // 3. Identify clones
  IntSet observableMutations;

  typedef std::map<std::pair<int, IntSet>, double> CloneFrequencyMap; // define cloneFrequencyMap as {(s, X): freq}
  CloneFrequencyMap cloneFrequencyMap;  // init cloneFrequencyMap

  // *define cloneCellMap as {(s, X): cell_ids}
  typedef std::map<std::pair<int, IntSet>, IntVector> CloneCellMap;
  CloneCellMap cloneCells;

  for (int s = 0; s < _nrAnatomicalSites; ++s)  // loop over each site
  {
    if (!_isActiveAnatomicalSite[s]) continue;  // make sure it's an active site
    DoubleVector freq_s;
    getMutationFrequencies(freq_s, s);  // 1. get mutation freqs at the site
    
    for (int i = 0; i < _nrMutations; ++i)  // loop over each mutation
    {
      if (freq_s[i] >= _mutFreqThreshold) // mutation freq higher threshold
      {
        observableMutations.insert(i);  // 2. get observable mutations
      }
    }
    
    for (const auto& kv : _extantCellsByDrivers[s]) // loop over cells at the site
    {
      const IntSet& driverMutations = kv.first; // get driver muts for the cells
      for (const Cell& cell : kv.second)  // loop over each cell
      {
        IntSet mutations(cell.getPassengerMutations().begin(),  
                         cell.getPassengerMutations().end()); // get pass muts for the cell
        mutations.insert(driverMutations.begin(),
                         driverMutations.end());  // add driver muts for the cell
        
        IntSet X;
        
        std::set_intersection(observableMutations.begin(),
                              observableMutations.end(),
                              mutations.begin(),
                              mutations.end(),
                              std::inserter(X, X.begin())); // get observable muts X for the cell
        
        // 3. a clone contains cells with same observable muts X at site s.
        if (cloneFrequencyMap.count(std::make_pair(s, X)) == 0) // no clone at site s with mut X
        {
          cloneFrequencyMap[std::make_pair(s, X)] = 0;  // init freq = 0
        }
        ++cloneFrequencyMap[std::make_pair(s, X)];  // freq + 1

        // *add cells to each clone
        if (cloneCells.count(std::make_pair(s, X)) == 0)
        {
          cloneCells[std::make_pair(s, X)] = std::vector<int>();
        }
        cloneCells[std::make_pair(s, X)].push_back(cell.getID());
      }
    }
    
    // 4. Compute clone frequencies
    int nrExtantCells_s = getNrExtantCells(s);  // get the cell number at site s
    for (auto& kv : cloneFrequencyMap)  // loop over clones
    {
      if (kv.first.first == s)  // current site
      {
        kv.second /= nrExtantCells_s; // 4. normalize clone count by total number of cells
      }
    }
  }
  
  // *output cells each clone
  std::ofstream clone_outfile("cloneCells.txt");
  //std::cout << "Total clones: " << cloneCells.size() << std::endl;
  clone_outfile << "Total clones: " << cloneCells.size() << std::endl;

  for (const auto& kv: cloneCells)
  {
    //std::cout << "Clone size: " << kv.second.size() << std::endl;
    clone_outfile << "Clone size: " << kv.second.size() << std::endl;

    int s = kv.first.first;
    if (!_isActiveAnatomicalSite[s]) continue;
          
    if (kv.second.size() > 100)
    {
      const IntSet& driverMutations = kv.first.second;
      //std::cout << s << "\t{";
      clone_outfile << s << "\t{";
      for (int i : driverMutations)
      {
        //std::cout << i << ",";
        clone_outfile << i << ",";
      }
      //std::cout << "}\t";
      clone_outfile << "}\t";
      for (int j : kv.second)
      {
        //std::cout << j << ",";
        clone_outfile << j << ",";
      }
      //std::cout << std::endl;
      clone_outfile << std::endl;
    }
  }
  clone_outfile.close();
  
  // construct perfect phylogeny matrix
  _observableMutationToCellTreeMutation = IntVector(observableMutations.size());
  _cellTreeMutationToObservableMutation.clear();
  int j = 0;
  for (int i : observableMutations) // loop over observable muts i
  {
    _observableMutationToCellTreeMutation[j] = i; // reorder i from 0
    _cellTreeMutationToObservableMutation[i] = j;
    
    if (_driverMutations.count(i) == 1) // i is driver
    {
      std::cout << "Mutation " << j << " is a driver mutation" << std::endl;
    }
    
    ++j;
  }
  
  // matrix (clone x muts)
  BoolMatrix matrix(cloneFrequencyMap.size(), // clone #
                    BoolVector(_observableMutationToCellTreeMutation.size(), false)); // Muts #
  IntVector anatomicalSiteMap(cloneFrequencyMap.size(), 0); // clone #
  int i = 0;
  for (auto& kv : cloneFrequencyMap)  // loop over clones
  {
    const int s = kv.first.first; // get s
    const IntSet& mutationSet = kv.first.second;  // get X
    for (int j : mutationSet) // loop over X
    {
      int jj = _cellTreeMutationToObservableMutation[j];  // get observable mut #
      matrix[i][jj] = true; // set clone i observation mut
    }
    
    anatomicalSiteMap[i] = s; // set clone i site
    ++i;  // next clone
  }
  
  IntSetVector toOriginalColumns;
  sortAndCondense(matrix, toOriginalColumns);
  
  Digraph newT;
  Node newRoot = lemon::INVALID;
  IntNodeMap leafToTaxa(newT);
  IntArcMap arcToCharacter(newT);
  IntSetNodeMap nodeToMutations(newT);
  constructTree(matrix,
                newT, newRoot,
                leafToTaxa, arcToCharacter, nodeToMutations);
  
  IntNodeMap anatomicalSite(newT, 0);
  StringNodeMap anatomicalSiteStr(newT, "");
  StringNodeMap label(newT, "");
  StringToNodeMap invMap;
  IntSetNodeMap stateVector(newT);
  constructMaps(newT,
                newRoot,
                leafToTaxa,
                anatomicalSiteMap,
                _anatomicalSiteLabel,
                arcToCharacter,
                nodeToMutations,
                toOriginalColumns,
                anatomicalSite,
                anatomicalSiteStr,
                label,
                invMap,
                stateVector);
  
  // clone freq for each node
  DoubleNodeMap cloneFrequency(newT, 0);  // nodes #
  for (NodeIt v(newT); v != lemon::INVALID; ++v)  // loop over nodes v
  {
    IntSet mutations;
    for (int j : nodeToMutations[v])  // loop over muts j in v
    {
      stateVector[v].insert(toOriginalColumns[j].begin(), toOriginalColumns[j].end());  // original muts for v
      for (int jj : toOriginalColumns[j])
        mutations.insert(_observableMutationToCellTreeMutation[jj]);  // original observable muts
    }
    if (OutArcIt(newT, v) == lemon::INVALID)  // leaves
    {
      const int s = anatomicalSiteMap[leafToTaxa[v]]; // get site s
      assert(cloneFrequencyMap.count(std::make_pair(s, mutations)) == 1);
      
      cloneFrequency[v] = cloneFrequencyMap[std::make_pair(s, mutations)];  // add clone freq
    }
  }
  
  delete _pAnatomicalSiteProportions;
  delete _pMutations;
  delete _pAnatomicalSite;
  delete _pSampleProportions;
  delete _pCloneT;
  
  _pCloneT = new CloneTree(newT, newRoot, label, anatomicalSiteStr);
  _pAnatomicalSiteProportions = new DoubleNodeMap(_pCloneT->tree(), 0);
  _pSampleProportions = new DoubleVectorNodeMap(_pCloneT->tree(), DoubleVector(_nrSamplesPerAnatomicalSite, 0));
  _pMutations = new IntSetNodeMap(_pCloneT->tree(), IntSet());
  _pAnatomicalSite = new IntNodeMap(_pCloneT->tree(), -1);
  for (NodeIt v(_pCloneT->tree()); v != lemon::INVALID; ++v)  // loop over nodes v
  {
    assert(invMap.count(_pCloneT->label(v)) == 1);
    Node vv = invMap[_pCloneT->label(v)];
    _pMutations->set(v, stateVector[vv]);

    if (_pCloneT->isLeaf(v))  // v is leave
    {
      _pAnatomicalSiteProportions->set(v, cloneFrequency[vv]);
      _pAnatomicalSite->set(v, anatomicalSiteMap[leafToTaxa[vv]]);
    }
  }
}

// label nodes
void Simulation::constructMaps(const Digraph& newT,
                               const Node& newRoot,
                               const IntNodeMap& leafToTaxa,
                               const IntVector& anatomicalSiteMap,
                               const StringVector& anatomicalSiteLabel,
                               const IntArcMap& arcToCharacter,
                               const IntSetNodeMap& nodeToMutations,
                               const IntSetVector& toOriginalColumns,
                               IntNodeMap& anatomicalSite,
                               StringNodeMap& anatomicalSiteStr,
                               StringNodeMap& label,
                               StringToNodeMap& invMap,
                               IntSetNodeMap& stateVector)
{
  char buf[1024];

  for (NodeIt v(newT); v != lemon::INVALID; ++v)  // loop over nodes v
  {
    if (v != newRoot) // not root
    {
      label[v] = "";
      if (arcToCharacter[InArcIt(newT, v)] != -1) // incoming arc not -1 (not root)
      {
        for (int j : toOriginalColumns[arcToCharacter[InArcIt(newT, v)]]) // loop over original cols
        {
          if (label[v] != "")
            label[v] += ";";  // seperate by ;
          snprintf(buf, 1024, "%d", j);
          label[v] += buf;  // add j as label
        }
        invMap[label[v]] = v; // {label:node}
      }
    }
    else
    {
      label[v] = "GL";  // root label
      invMap[label[v]] = v;
    }
    if (OutArcIt(newT, v) == lemon::INVALID)  // leaf
    {
      const int s = anatomicalSiteMap[leafToTaxa[v]]; // site

      anatomicalSite[v] = s;  // site
      anatomicalSiteStr[v] = anatomicalSiteLabel[s];  // site label
    }
  }
  
  for (NodeIt v(newT); v != lemon::INVALID; ++v)  // loop over nodes v
  {
    if (OutArcIt(newT, v) == lemon::INVALID)  // leaf
    {
      label[v] = label[newT.source(InArcIt(newT, v))];  // label as parent
      label[v] += "_" + anatomicalSiteStr[v]; // add site to label
      invMap[label[v]] = v;
    }
  }
}

void Simulation::writeDOT(const Digraph& T,
                          Node root,
                          const IntSetArcMap& mutations,
                          const DoubleNodeMap& cloneFrequency,
                          const IntNodeMap& anatomicalSite,
                          std::ostream& out) const
{
  out << "digraph T {" << std::endl;
  
  for (NodeIt v(T); v != lemon::INVALID; ++v)
  {
    if (OutArcIt(T, v) != lemon::INVALID)
    {
      out << "\t" << T.id(v) << " [label=\"\"]" << std::endl;
    }
    else
    {
      out << "\t" << T.id(v) << " [label=\"" << cloneFrequency[v] << "\\n" << anatomicalSite[v] << "\"]" << std::endl;
    }
  }
  
  for (ArcIt a(T); a != lemon::INVALID; ++a)
  {
    out << "\t" << T.id(T.source(a)) << " -> " << T.id(T.target(a)) << " [label=\"";
    const IntSet& mutations_a = mutations[a];
    bool first = true;
    for (int j : mutations_a)
    {
      if (first)
      {
        first = false;
      }
      else
      {
        out << " ";
      }
      out << j;
    }
    out << "\"]" << std::endl;
  }
  
  out << "}" << std::endl;
}

// construct clone tree based on boolean matrix (clones x observable mutations)
void Simulation::constructTree(const BoolMatrix& matrix,
                               Digraph& T,
                               Node& root,
                               IntNodeMap& leafToTaxa,
                               IntArcMap& arcToCharacter,
                               IntSetNodeMap& nodeToMutations)
{
  const int m = matrix.size();  // rows
  assert(m > 0);
  const int n = matrix.front().size();  // cols
  
  T.clear();
  root = T.addNode();
  nodeToMutations[root] = IntSet();
  
  for (int i = 0; i < m; ++i)  // loop over rows
  {
    Node parent = root;
    for (int j = 0; j < n; ++j) // loop over cols
    {
      if (matrix[i][j]) // true: edge present
      {
        bool found = false;
        for (OutArcIt a(T, parent); a != lemon::INVALID; ++a) // loop over arcs from source node parent
        {
          if (arcToCharacter[a] == j) // arc j exists
          {
            parent = T.target(a); // update target node as parent
            found = true;
            break;
          }
        }
        
        if (!found)
        {
          Node child = T.addNode();
          Arc a = T.addArc(parent, child);  // add arc between parent child
          arcToCharacter[a] = j;  // set arc as j
          nodeToMutations[child] = nodeToMutations[parent]; // child has the same mut as the parent
          nodeToMutations[child].insert(j); // add new mut at j
          parent = child; // update child as parent
        }
      }
    }
    
    // attach leaf
    Node leaf = T.addNode();
    Arc a = T.addArc(parent, leaf); // create arc between parent and leaf
    arcToCharacter[a] = -1; // set extant arc as -1
    leafToTaxa[leaf] = i; // label leaf as clone # i
    nodeToMutations[leaf] = nodeToMutations[parent];  // leaf has the same muts as parent
  }
}

// sort matrix by count of 1s at each col; remove duplicated cols.
void Simulation::sortAndCondense(BoolMatrix& matrix,
                                 IntSetVector& originalColumns)
{
  const int m = matrix.size();  // rows
  assert(m > 0);
  const int n = matrix.front().size();  // cols
  
  // 1. count number of 1s for every column and transpose matrix
  IntVector I(matrix.front().size(), 0);
  BoolMatrix transpose(n, BoolVector(m, false));
  for (int i = 0; i < m; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      transpose[j][i] = matrix[i][j]; // transpose
      if (matrix[i][j])
        ++I[j];  // count 1s at col j
    }
  }
  
  typedef std::map<BoolVector, std::pair<int, IntSet> > BoolVectorMap;  // {col:(count 1s, col # j)}
  BoolVectorMap uniqueColumns;
  for (int j = 0; j < n; ++j) // loop over cols
  {
    if (uniqueColumns.count(transpose[j]) == 0) // create new col
    {
      uniqueColumns[transpose[j]] = std::make_pair(I[j], IntSet());
    }
    uniqueColumns[transpose[j]].second.insert(j); // add to existing col
  }
  
  BoolMatrix newMatrix(m);
  while (!uniqueColumns.empty())  // when uniqueCols not empty
  {
    // find largest unique columns
    int max = -1;
    for (const auto& kv : uniqueColumns)
    {
      const int nrOf1s = kv.second.first; // count 1s
      if (nrOf1s > max)
      {
        max = nrOf1s; // max count 1s
      }
    }
    
    for (const auto& kv : uniqueColumns)
    {
      const BoolVector& column = kv.first;  // col
      const int nrOf1s = kv.second.first; // count 1s
      const IntSet& org = kv.second.second; // original col # j
      
      if (nrOf1s == max)
      {
        for (int i = 0; i < m; ++i)
        {
          newMatrix[i].push_back(column[i]);  // add col with max 1s to new matrix
        }
        
        originalColumns.push_back(org); // original col # j
        uniqueColumns.erase(column);  // remove col from uniqueCol
        break;
      }
    }
  }
  
  matrix = newMatrix;
}

void Simulation::getMutationFrequencies(DoubleVector& freq,
                                        int s) const
{
  assert(0 <= s && s < _nrAnatomicalSites);
  
  freq = DoubleVector(_nrMutations, 0);
  for (const auto& kv : _extantCellsByDrivers[s]) // loop over cells at s
  {
    const IntSet& driverMutations = kv.first; // driver muts
    for (const Cell& cell : kv.second)  // loop over cells
    {
      for (int i : cell.getPassengerMutations())  // loop over pass
      {
        ++freq[i];  // add each pass
      }
      for (int i : driverMutations) // loop over driver
      {
        ++freq[i];  // add each driver
      }
    }
  }
  
  int nrExtantCells_s = getNrExtantCells(s);  // get total # of cells at s
  for (int i = 0; i < _nrMutations; ++i)  // loop over muts
  {
    freq[i] /= nrExtantCells_s; // normalize each mut # by cell #
  }
}

bool Simulation::constructSampledCloneTree()
{
  _sampledMutations.clear();
  _sampledDriverMutations.clear();
  _sampledMutationsByAnatomicalSite = IntSetVector(_nrAnatomicalSites);
  const int n = getNrMutations();
  for (int i = 0; i < n; ++i)
  {
    bool sampled_i = false;
    for (int s = 0; s < _nrAnatomicalSites && !sampled_i; ++s)
    {
      if (!_isActiveAnatomicalSite[s]) continue;
      const int k_s = s == 0 ? _nrSamplesPrimary : _nrSamplesPerAnatomicalSite;
      for (int p = 0; p < k_s; ++p)
      {
        if (g_tol.nonZero(_freq[s][p][i]))
        {
          _sampledMutations.insert(i);
          _sampledMutationsByAnatomicalSite[s].insert(i);
        }
      }
    }
  }
  
  //*If no leaves were sampled, there is nothing to construct
  if (_sampledLeaves.empty())
  {
    return false;
  }
  
  Digraph newT;
  Node newRoot = lemon::INVALID;
  StringNodeMap anatomicalSiteStr(newT, "");
  StringNodeMap label(newT, "");

  BoolNodeMap filterNodes(_pCloneT->tree(), false);
  BoolArcMap filterArcs(_pCloneT->tree(), false);
  SubDigraph subT(_pCloneT->tree(), filterNodes, filterArcs);
  for (Node v : _sampledLeaves)
  {
    filterNodes[v] = true;
    InArcIt a(_pCloneT->tree(), v);
    while (a != lemon::INVALID)
    {
      Node u = _pCloneT->tree().source(a);
      filterNodes[u] = true;
      filterArcs[a] = true;
      a = InArcIt(_pCloneT->tree(), u);
    }
  }
  
  /// The parameter should be a map, whose key type
  /// is the Node type of the destination digraph, while the value type is
  /// the Node type of the source digraph.
  NodeNodeMap ref(newT);
  
  IntSetNodeMap newMutationMap(newT);
  lemon::digraphCopy(subT, newT)
    .node(_pCloneT->root(), newRoot)
    .nodeMap(_pCloneT->getIdMap(), label)
    .nodeMap(*_pMutations, newMutationMap)
    .nodeMap(_pCloneT->getLeafLabeling(), anatomicalSiteStr)
    .nodeCrossRef(ref)
    .run();
  
  std::map<std::string, IntSet> labelToMutations;
  for (NodeIt v(newT); v != lemon::INVALID; ++v)
  {
    labelToMutations[label[v]] = newMutationMap[v];
  }
  
  // Contract degree one nodes
  bool changed = true;
  while (changed)
  {
    changed = false;
    for (NodeIt v(newT); v != lemon::INVALID; ++v)
    {
      if (v != newRoot && lemon::countOutArcs(newT, v) == 1)
      {
        Node u = newT.source(InArcIt(newT, v));
        Node w = newT.target(OutArcIt(newT, v));
        if (lemon::countOutArcs(newT, w) != 0)
        {
          if (labelToMutations.count(label[w]) == 1)
          {
            labelToMutations.erase(label[w]);
          }
          
          label[w] = label[v] + ";" + label[w];
          newMutationMap[w].insert(newMutationMap[v].begin(), newMutationMap[v].end());
          labelToMutations[label[w]] = newMutationMap[w];
          
          if (labelToMutations.count(label[v]) == 1)
          {
            labelToMutations.erase(label[v]);
          }
          
          newT.erase(v);
          
          newT.addArc(u, w);
          changed = true;
          break;
        }
      }
    }
  }
  
  // Add migration vertices
  Digraph::ArcMap<NodeSet> arcGtoVerticesNewT(_G);
  for (NodeIt v(newT); v != lemon::INVALID; ++v)
  {
    if (lemon::countOutArcs(newT, v) == 0)
    {
      Node vv = ref[v];
      const int t = (*_pAnatomicalSite)[vv];
      Node v_t = _indexToVertexG[t];
      
      // find largest intersection
      Arc migrationArc = lemon::INVALID;
      int overlap = 0;
      
      for (InArcIt a(_G, v_t); a != lemon::INVALID; ++a)
      {
        IntSet X = getMutationsOfMigrationEdge(a);
        IntSet diff, diff2;
        std::set_difference(X.begin(), X.end(),
                            (*_pMutations)[vv].begin(),
                            (*_pMutations)[vv].end(),
                            std::inserter(diff, diff.begin()));

        // if cell is ancestor of v_i then v_i must contain all mutations of cell
        if (X.size() > overlap && diff.empty())
        {
          migrationArc = a;
          overlap = X.size();
        }
      }
      
      if (migrationArc != lemon::INVALID)
      {
        arcGtoVerticesNewT[migrationArc].insert(v);
      }
    }
  }
  
  Digraph::ArcMap<NodeSet> arcGtoCumVerticesNewT(_G);
  Digraph::ArcMap<Arc> parentArcMap(_G, lemon::INVALID);
  if (!computeParent(arcGtoVerticesNewT, parentArcMap))
  {
    return false;
  }
  computeCumVerticesNewT(arcGtoVerticesNewT, parentArcMap, arcGtoCumVerticesNewT, _rootG);
  
  std::map<std::string, std::string> labelToAnatomicalSite;
  char buf[1024];
  IntVector subLabelIndex(_nrAnatomicalSites, 0);
  StringNodeMap anatomicalSiteNewT(newT);
  BoolNodeMap isMutationNode(newT, false);
  for (ArcIt a(_G); a != lemon::INVALID; ++a)
  {
    const int t = _anatomicalSiteMap[_G.target(a)];
    const NodeSet& leaves = arcGtoCumVerticesNewT[a];
    if (leaves.size() > 0)
    {
      IntSet X_a = getMutationsOfMigrationEdge(a);
      Node seedingNode;
      int overlap = std::numeric_limits<int>::max();
      for (NodeIt v(newT); v != lemon::INVALID; ++v)
      {
        if (OutArcIt(newT, v) == lemon::INVALID) continue;
        
        const IntSet& mut_v = newMutationMap[v];
        IntSet diff;
        
        std::set_difference(X_a.begin(), X_a.end(),
                            mut_v.begin(), mut_v.end(),
                            std::inserter(diff, diff.begin()));
        
        if (mut_v.size() < overlap && diff.empty())
        {
          seedingNode = v;
          overlap = mut_v.size();
        }
      }
      
      NodeSet affected_children_of_seedingNode;
      for (OutArcIt aa(newT, seedingNode); aa != lemon::INVALID; ++aa)
      {
        Node w = newT.target(aa);
        for (Node leaf : leaves)
        {
          if (BaseTree::isAncestor(newT, w, leaf))
          {
            affected_children_of_seedingNode.insert(w);
            break;
          }
        }
      }
      
      Node vv = newT.addNode();
      isMutationNode[vv] = true;
      anatomicalSiteNewT[vv] = _anatomicalSiteLabel[t];
      snprintf(buf, 1024, "%s_%d",
               _anatomicalSiteLabel[t].c_str(),
               subLabelIndex[t]++);
      label[vv] = buf;
      labelToMutations[label[vv]] = labelToMutations[label[seedingNode]];
      labelToAnatomicalSite[label[vv]] = _anatomicalSiteLabel[t];
      newT.addArc(seedingNode, vv);
      for (Node w : affected_children_of_seedingNode)
      {
        newT.erase(InArcIt(newT, w));
        newT.addArc(vv, w);
      }
    }
  }
  
  // Contract degree one nodes
  changed = true;
  while (changed)
  {
    changed = false;
    for (NodeIt v(newT); v != lemon::INVALID; ++v)
    {
      if (v != newRoot && lemon::countOutArcs(newT, v) == 1)
      {
        Node u = newT.source(InArcIt(newT, v));
        Node w = newT.target(OutArcIt(newT, v));
        if (lemon::countOutArcs(newT, w) != 0)
        {
          if (isMutationNode[w])
          {
            std::string sStr = anatomicalSiteNewT[w];
            std::string label_v = label[v];
            label[w] = label_v;
            labelToAnatomicalSite[label[w]] = sStr;
          }
          else
          {
            assert(isMutationNode[v]);
            std::string sStr = anatomicalSiteNewT[v];
            std::string label_w = label[w];
            anatomicalSiteNewT[w] = sStr;
            labelToAnatomicalSite[label[w]] = sStr;
          }
          
          newT.addArc(u, w);
          newT.erase(v);

          changed = true;
          break;
        }
      }
    }
  }
  
  _pCloneTT = new CloneTree(newT, newRoot, label, anatomicalSiteStr);
  _pSampleProportionsTT = new DoubleVectorNodeMap(_pCloneTT->tree(), DoubleVector(_nrSamplesPerAnatomicalSite, 0));
  _pMutationsTT = new IntSetNodeMap(_pCloneTT->tree(), IntSet());
  _pAnatomicalSiteTT = new IntNodeMap(_pCloneTT->tree(), -1);
  _pVertexLabelingTT = new StringNodeMap(_pCloneTT->tree());
  for (NodeIt v(_pCloneTT->tree()); v != lemon::INVALID; ++v)
  {
    const std::string& label_v = _pCloneTT->label(v);
    assert(labelToMutations.count(label_v) == 1);
    _pMutationsTT->set(v, labelToMutations[label_v]);
    if (_pCloneTT->isLeaf(v))
    {
      Node vv = _pCloneT->getNodeByLabel(label_v);
      assert(vv != lemon::INVALID);
      _pSampleProportionsTT->set(v, (*_pSampleProportions)[vv]);
      _pAnatomicalSiteTT->set(v, (*_pAnatomicalSite)[vv]);
      _pVertexLabelingTT->set(v, _anatomicalSiteLabel[(*_pAnatomicalSite)[vv]]);
    }
    else if (labelToAnatomicalSite.count(label_v) == 1)
    {
      _pVertexLabelingTT->set(v, labelToAnatomicalSite[label_v]);
    }
  }
  
  // match leaves with seeding cells
  for (Node v_i : _pCloneTT->leafSet())
  {
    const int t = (*_pAnatomicalSiteTT)[v_i];
    Node v_t = _indexToVertexG[t];
    _pVertexLabelingTT->set(v_i, _anatomicalSiteLabel[t]);

    // find largest intersection
    Arc migrationArc = lemon::INVALID;
    int overlap = 0;
    
    for (InArcIt a(_G, v_t); a != lemon::INVALID; ++a)
    {
      IntSet X = getMutationsOfMigrationEdge(a);
      _arcGtoSampledMutations[a] = X;
      IntSet diff;
      std::set_difference(X.begin(), X.end(),
                          (*_pMutationsTT)[v_i].begin(), (*_pMutationsTT)[v_i].end(),
                          std::inserter(diff, diff.begin()));
      // if cell is ancestor of v_i then v_i must contain all mutations of cell
      if (X.size() > overlap && diff.empty())
      {
        migrationArc = a;
        overlap = X.size();
      }
    }
    
    if (migrationArc != lemon::INVALID)
    {
      _arcGtoVerticesTT[migrationArc].insert(v_i);
    }
  }
  
  // Find right order of arcs, reverse BFS order
//  IntNodeMap dist(_pCloneTT->tree(), -1);
//  lemon::bfs(_pCloneTT->tree()).distMap(dist).run(_pCloneTT->root());
//  int max_dist = lemon::mapMaxValue(_pCloneTT->tree(), dist);
//  ArcVector arcs;
//  while (max_dist > 0)
//  {
//    for (NodeIt v_i(_pCloneTT->tree()); v_i != lemon::INVALID; ++v_i)
//    {
//      if (dist[v_i] == max_dist)
//      {
//        Arc inArc = InArcIt(_pCloneTT->tree(), v_i);
//        assert(inArc != lemon::INVALID);
//        arcs.push_back(inArc);
//      }
//    }
//    --max_dist;
//  }
//  
//  // construct vertex labeling
//  for (ArcIt aa(_G); aa != lemon::INVALID; ++aa)
//  {
//    if (_arcGtoVerticesTT[aa].empty()) continue;
//    const IntSet& seeding_muts = _arcGtoSampledMutations[aa];
//    Node v_t = _G.target(aa);
//    const int t = _anatomicalSiteMap[v_t];
//    
//    // find matching migration edge in T
//    for (Arc a : arcs)
//    {
//      Node v_i = _pCloneTT->tree().source(a);
//      Node v_j = _pCloneTT->tree().target(a);
//      
//      if (v_i == _pCloneTT->root()) continue;
//      
//      const IntSet& muts_i = (*_pMutationsTT)[v_i];
//      const IntSet& muts_j = (*_pMutationsTT)[v_j];
//      
//      IntSet tmp1;
//      std::set_difference(seeding_muts.begin(), seeding_muts.end(),
//                          muts_i.begin(), muts_i.end(),
//                          std::inserter(tmp1, tmp1.begin()));
//      
//      IntSet tmp2;
//      std::set_difference(seeding_muts.begin(), seeding_muts.end(),
//                          muts_j.begin(), muts_j.end(),
//                          std::inserter(tmp2, tmp2.begin()));
//      
//      if (!tmp1.empty() && tmp2.empty())
//      {
////        if (!(*_pVertexLabelingTT)[v_j].empty())
////        {
////          _inNeedOfPolytomyResolution = true;
////        }
//        _pVertexLabelingTT->set(v_j, _anatomicalSiteLabel[t]);
//      }
//    }
//  }
  
  _pVertexLabelingTT->set(_pCloneTT->root(), _anatomicalSiteLabel[0]);
  finishLabeling(*_pCloneTT, _pCloneTT->root());
  
  return true;
}

// get the largest parent arc
bool Simulation::computeParent(const Digraph::ArcMap<NodeSet>& arcsGtoVerticesNewT,
                               Digraph::ArcMap<Arc>& parentArcMap) const
{
  for (ArcIt a(_G); a != lemon::INVALID; ++a) // loop over arcs a
  {
    if (arcsGtoVerticesNewT[a].empty()) continue; // no vertix, skip
    Node v_s = _G.source(a);  // source node of a
    if (v_s == _rootG) continue;  // root node has no parent, skip
    
    IntSet X_a = getMutationsOfMigrationEdge(a);  // get muts on edge a
    int overlap = 0;
    Arc parent = lemon::INVALID;
    for (InArcIt aa(_G, v_s); aa != lemon::INVALID; ++aa) // loop over incoming arcs aa to node v_s
    {
      IntSet X_aa = getMutationsOfMigrationEdge(aa);  // get muts on aa
      IntSet diff;
      std::set_difference(X_aa.begin(), X_aa.end(),
                          X_a.begin(), X_a.end(),
                          std::inserter(diff, diff.begin())); // muts in X_aa but not in X_a
      if (X_aa.size() > overlap && diff.empty())  //  larger X_aa is a subset of X_a
      {
        parent = aa;  // update parent arc
        overlap = X_aa.size();  // update X
      }
    }
    if (parent == lemon::INVALID) // no parent
    {
      return false;
    }
    parentArcMap[a] = parent; // set parent arc of a
  }
  
  return true;
}

// get all reachable vertices from arc
void Simulation::computeCumVerticesNewT(const Digraph::ArcMap<NodeSet>& arcsGtoVerticesNewT,
                                        const Digraph::ArcMap<Arc>& parentArcMap,
                                        Digraph::ArcMap<NodeSet>& arcsGtoCumVerticesNewT,
                                        Node v_s) const
{
  if (OutArcIt(_G, v_s) == lemon::INVALID)  // no outcoming arc from v_s (leaf)
  {
    for (InArcIt a(_G, v_s); a != lemon::INVALID; ++a)  // loop over incoming arcs to v_s
    {
      arcsGtoCumVerticesNewT[a] = arcsGtoVerticesNewT[a]; // add vertices
    }
  }
  else
  {
    for (OutArcIt a(_G, v_s); a != lemon::INVALID; ++a) // loop over outcoming arcs from v_s
    {
      if (arcsGtoVerticesNewT[a].empty()) continue; // no vertex, skip
      
      IntSet X_a = getMutationsOfMigrationEdge(a);  // get X_a for a
      
      Node v_t = _G.target(a);  // get target node v_t of a
      arcsGtoCumVerticesNewT[a] = arcsGtoVerticesNewT[a]; // add vertices
      
      for (OutArcIt aa(_G, v_t); aa != lemon::INVALID; ++aa)  // loop over outcoming arcs from v_t
      {
        if (arcsGtoVerticesNewT[aa].empty()) continue; // no vertex, skip
        if (a == parentArcMap[aa])  // a is parent of aa
        {
          // recurse
          computeCumVerticesNewT(arcsGtoVerticesNewT, parentArcMap, arcsGtoCumVerticesNewT, v_t); // recursion
          arcsGtoCumVerticesNewT[a].insert(arcsGtoCumVerticesNewT[aa].begin(),
                                           arcsGtoCumVerticesNewT[aa].end()); // add aa to a
        }
      }
    }
  }
}

void Simulation::finishLabeling(const CloneTree& T, Node v_i)
{
  // unlabeled
  if ((*_pVertexLabelingTT)[v_i].empty())
  {
    assert(v_i != T.root());
    Node parent = T.tree().source(InArcIt(T.tree(), v_i));
    
    assert(!(*_pVertexLabelingTT)[parent].empty());
    
    _pVertexLabelingTT->set(v_i, (*_pVertexLabelingTT)[parent]);
  }
  
  for (OutArcIt a(T.tree(), v_i); a != lemon::INVALID; ++a)
  {
    Node v_j = T.tree().target(a);
    finishLabeling(T, v_j);
  }
}

void Simulation::writeObservedClustering(std::ostream& out) const
{
  assert(_pCloneTT);
  const Digraph& T = _pCloneTT->tree();
  for (NodeIt v(T); v != lemon::INVALID; ++v)
  {
    if (v == _pCloneTT->root() || _pCloneTT->isLeaf(v))
      continue;
    
    out << _pCloneTT->label(v) << std::endl;
  }
}
