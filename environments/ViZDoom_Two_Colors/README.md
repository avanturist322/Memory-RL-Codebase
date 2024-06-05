# ViZDoom-Two-Colors Environment

* environments/ViZDoom_Two_Colors/env - environment files & pretrained a2c agent to collect trajectories
* environments/VizDoom_Two_colors/env_configs - ViZDoom-Two-Colors environments configurations:

.
├── ...
├── environments/                    
│   ├── ViZDoom_Two_Colors/          
│       ├── env/                                    # environment files & pretrained A2C agent to collect trajectores
│       └── env_config/                             # environment configurations
|           ├── custom_scenario_no_pil000.cfg       # without pillar
|           └── custom_scenario000.cfg              # with dissappearing pillar
|   ├── ...
|   └── ...
└── ...

├── app
│   ├── css
│   │   ├── **/*.css
│   ├── favicon.ico
│   ├── images
│   ├── index.html
│   ├── js
│   │   ├── **/*.js
│   └── partials/template
├── dist (or build)
├── node_modules
├── bower_components (if using bower)
├── test
├── Gruntfile.js/gulpfile.js
├── README.md
├── package.json
├── bower.json (if using bower)
└── .gitignore
