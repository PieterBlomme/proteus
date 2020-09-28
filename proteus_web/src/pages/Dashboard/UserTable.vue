<template>

  <base-table :data="data"
              :columns="columns"
              thead-classes="text-primary">
  </base-table>
</template>
<script>
  import { BaseTable } from "@/components";
  import axios from 'axios';
  export default {
    components: {
      BaseTable
    },
    computed: {
      table(){
        return this.data;
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
  }
</script>
<style>
</style>
