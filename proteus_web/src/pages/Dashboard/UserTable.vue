<template>
  <table class="table tablesorter" :class="tableClass">
    <thead :class="theadClasses">
    <tr>
      <slot name="columns">
        <th v-for="column in columns" :key="column">{{column}}</th>
      </slot>
    </tr>
    </thead>
    <tbody :class="tbodyClasses">
    <tr v-for="(item, index) in data" :key="index">
      <slot :row="item">
        <td>{{itemValue(item, 'name')}}</td>
        <td><base-button  @click="toggleState(item)" >{{stateValue(item)}}</base-button></td>

      </slot>
    </tr>
    </tbody>
  </table>
</template>
<script>
  import { BaseButton } from "@/components";
  import axios from 'axios';
  export default {
    components: {
      BaseButton
    },
    props: {
      type: {
        type: String, // striped | hover
        default: "",
        description: "Whether table is striped or hover type"
      },
      theadClasses: {
        type: String,
        default: '',
        description: "<thead> css classes"
      },
      tbodyClasses: {
        type: String,
        default: '',
        description: "<tbody> css classes"
      }
    },
    computed: {
      table(){
        return this.data;
      },
      tableClass() {
        return this.type && `table-${this.type}`;
      }
    },
    data() {
      return {
        data: null,
        columns: ['name', 'state']
      }
    },
    mounted() {
      axios
        .get('http://localhost:8080/models')
        .then(response => {
          this.data = response.data
        })
    },
    methods: {
      hasValue(item, column) {
        return item[column.toLowerCase()] !== "undefined";
      },
      stateColumn(column) {
        console.log('hello there');
        console.log(column.toLowerCase())
        return column.toLowerCase() == "state";
      },
      itemValue(item, column) {
        return item[column.toLowerCase()];
      },
      toggleState(item) {
        const model = { name: item['name'] };
        if(item['state'] == 'READY')
        {
          axios
            .post('http://localhost:8080/unload/', model)
                .then(response => (console.log(response.url)))
        }
        else{
          axios
            .post('http://localhost:8080/load/', model)
                .then(response => (console.log(response.url)))
        }
      },
      stateValue(item, column) {
        if(item['state'] == 'READY')
        {
          return 'unload'
        }
        else{
          return 'load'
        }
      },
      
    }
  }
</script>
<style>
</style>
